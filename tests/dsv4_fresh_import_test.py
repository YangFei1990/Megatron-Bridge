"""
dsv4_fresh_import_test.py — Bridge validation: import, save, round-trip, forward.

TP=1, ETP=4 on GB200. Validates the Megatron Bridge for DeepSeek-V4-Flash.

Phases:
  1. Distributed init
  2. Import HF → Megatron via Bridge
  3. Save Megatron checkpoint
  4. Round-trip export: compare ALL exported HF weights against original
  5. Forward pass on 3 prompts (smoke test)
  6. Logit comparison vs FP8 reference (informational, not definitive)
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


MODEL_PATH = os.environ.get(
    "DSV4_MODEL_PATH", "/lustre/fsw/portfolios/coreai/users/weijiac/models/deepseek-ai/DeepSeek-V4-Flash"
)
CKPT_DIR = "/lustre/fsw/portfolios/coreai/users/weijiac/dsv4_flash_megatron_ckpt_tp1"
FP8_REF = "/lustre/fsw/portfolios/coreai/users/weijiac/dsv4_ref_logits.pt"
SAVE_OUT = "/lustre/fsw/portfolios/coreai/users/weijiac/dsv4_fresh_analysis.pt"

TP = int(os.environ.get("TP_SIZE", "1"))
ETP = int(os.environ.get("ETP_SIZE", "4"))

PROMPTS = [
    "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>What is 1+1?<\uff5cAssistant\uff5c></think>",
    "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>What is the capital of France?<\uff5cAssistant\uff5c></think>",
    "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>Write hello world in Python.<\uff5cAssistant\uff5c></think>",
]

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))


def log(msg):
    if rank == 0:
        print(msg, flush=True)


# ── Phase 1: Distributed init ─────────────────────────────────────────────
log("\n=== Phase 1: Distributed init ===")
torch.cuda.set_device(local_rank)
dist.init_process_group(
    "nccl",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    world_size=world_size,
    rank=rank,
)
from megatron.core import parallel_state, tensor_parallel


parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=TP,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=ETP,
)
tensor_parallel.model_parallel_cuda_manual_seed(42)
log(f"  TP={TP} ETP={ETP} world={world_size}")

# ── Phase 2: Import HF → Megatron ─────────────────────────────────────────
log("\n=== Phase 2: Import HF → Megatron ===")
t0 = time.time()

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.utils.common_utils import disable_mtp_for_inference


bridge = AutoBridge.from_hf_pretrained(MODEL_PATH)
hf_pretrained = PreTrainedCausalLM(MODEL_PATH)
provider = bridge._model_bridge.provider_bridge(hf_pretrained)
provider.tensor_model_parallel_size = TP
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 1
provider.expert_tensor_parallel_size = ETP
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
log(f"  Model created in {time.time() - t0:.1f}s")

t0 = time.time()
bridge.load_hf_weights(model)
dist.barrier()
log(f"  Weights loaded in {time.time() - t0:.1f}s")

for m in model:
    m.eval()
    disable_mtp_for_inference(m)

# ── Phase 3: Save Megatron checkpoint ──────────────────────────────────────
log("\n=== Phase 3: Save Megatron checkpoint ===")
os.makedirs(CKPT_DIR, exist_ok=True)
ckpt_path = os.path.join(CKPT_DIR, f"rank_{rank}.pt")
torch.save(model[0].state_dict(), ckpt_path)
dist.barrier()
log(f"  Saved to {CKPT_DIR}")

# ── Phase 4: Round-trip export — compare ALL weights ───────────────────────
log("\n=== Phase 4: Round-trip export (all weights) ===")
t0 = time.time()
export_count = 0
max_diffs = []
mismatched = []
for hf_name, tensor in bridge.export_hf_weights(model, show_progress=(rank == 0)):
    export_count += 1
    ref_tensor = hf_pretrained.state.get(hf_name)
    if ref_tensor is not None:
        diff = (tensor.float().cpu() - ref_tensor.float().cpu()).abs().max().item()
        max_diffs.append((hf_name, diff))
        if diff > 1e-4:
            mismatched.append((hf_name, diff))
dist.barrier()

if rank == 0:
    log(f"  Exported {export_count} tensors in {time.time() - t0:.1f}s")
    log(f"  Compared {len(max_diffs)} tensors against HF checkpoint")
    if mismatched:
        log(f"  WARNING: {len(mismatched)} tensors with max_diff > 1e-4:")
        for name, diff in mismatched[:10]:
            log(f"    {name}: {diff:.6e}")
    else:
        log(f"  ALL {len(max_diffs)} tensors match (max_diff <= 1e-4)")
    # Show top-5 largest diffs
    top5 = sorted(max_diffs, key=lambda x: -x[1])[:5]
    log("  Top-5 largest diffs:")
    for name, diff in top5:
        log(f"    {name}: {diff:.6e}")

# ── Phase 5: Forward pass (smoke) ─────────────────────────────────────────
log("\n=== Phase 5: Forward pass (smoke) ===")
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
fwd_bwd = get_forward_backward_func()


class _It:
    def __init__(self, ids, pos):
        self.b = {"tokens": ids, "position_ids": pos}
        self._d = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._d:
            raise StopIteration
        self._d = True
        return self.b


def _fwd(data_iterator, model, **_):
    b = next(data_iterator)
    return model(input_ids=b["tokens"], position_ids=b["position_ids"], attention_mask=None), lambda x, **k: x


def run_forward(prompt):
    ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
    with torch.no_grad():
        out = fwd_bwd(
            forward_step_func=_fwd,
            data_iterator=_It(ids, pos),
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=ids.shape[1],
            micro_batch_size=1,
            collect_non_loss_data=True,
        )
    logits_all = None
    if parallel_state.is_pipeline_last_stage() and isinstance(out, list) and out:
        t = out[0]
        ws = parallel_state.get_tensor_model_parallel_world_size()
        if ws > 1:
            gathered = [torch.zeros_like(t) for _ in range(ws)]
            dist.all_gather(gathered, t, group=parallel_state.get_tensor_model_parallel_group())
            logits_all = torch.cat(gathered, dim=2)[0].float().cpu()
        else:
            logits_all = t[0].float().cpu()
    return logits_all, ids[0].cpu().tolist()


results = {}
for pi, prompt in enumerate(PROMPTS):
    log(f"\n  Prompt {pi + 1}: {prompt[:60]}...")
    logits, token_ids = run_forward(prompt)
    if rank == 0 and logits is not None:
        seq_len = logits.shape[0]
        log(f"  seq_len={seq_len}, vocab={logits.shape[1]}")
        for pos in range(seq_len):
            top1 = logits[pos].argmax().item()
            log(f"    pos {pos:3d}: top1={tokenizer.decode([top1])!r:12s} [{top1:6d}]  std={logits[pos].std():.3f}")
        results[pi] = {"logits": logits, "token_ids": token_ids}
    dist.barrier()

# ── Phase 6: Logit analysis vs FP8 reference (informational) ──────────────
if rank == 0 and 0 in results and os.path.exists(FP8_REF):
    log("\n=== Phase 6: Logit analysis vs FP8 reference ===")
    log("  NOTE: This compares BF16 Megatron vs FP8 official inference.")
    log("  Differences reflect FP8/BF16 arithmetic drift, not Bridge bugs.")
    mg_last = results[0]["logits"][-1]
    ref = torch.load(FP8_REF, map_location="cpu", weights_only=True)
    fp8_last = ref["logits"].float()

    v = min(mg_last.shape[0], fp8_last.shape[0])
    mg, fp8 = mg_last[:v], fp8_last[:v]

    cos = F.cosine_similarity(mg.unsqueeze(0), fp8.unsqueeze(0)).item()
    log(f"  Cosine similarity:  {cos:.6f}")

    top5_mg = mg.topk(5)
    top5_fp8 = fp8.topk(5)
    log(
        f"  MG  top-5: {[(tokenizer.decode([i]), f'{v:.2f}') for i, v in zip(top5_mg.indices.tolist(), top5_mg.values.tolist())]}"
    )
    log(
        f"  FP8 top-5: {[(tokenizer.decode([i]), f'{v:.2f}') for i, v in zip(top5_fp8.indices.tolist(), top5_fp8.values.tolist())]}"
    )

    for K in [5, 10, 50]:
        mg_set = set(mg.topk(K).indices.tolist())
        fp8_set = set(fp8.topk(K).indices.tolist())
        jacc = len(mg_set & fp8_set) / len(mg_set | fp8_set)
        log(f"  Top-{K} Jaccard: {jacc:.4f}")

    torch.save({"results": {k: v for k, v in results.items()}, "cos_sim": cos}, SAVE_OUT)
    log(f"  Saved to {SAVE_OUT}")

log("\n=== All phases complete ===")
dist.destroy_process_group()
