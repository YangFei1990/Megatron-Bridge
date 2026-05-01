"""
dsv4_fresh_generate.py — Bridge validation: import + greedy generation.

TP=1, ETP=4 on GB200. Tests that the imported model produces coherent,
correct answers via auto-regressive generation.

3 prompts: math (17*13=221), reasoning (train problem), code (Fibonacci).
"""

import os
import time

import torch
import torch.distributed as dist


MODEL_PATH = os.environ.get(
    "DSV4_MODEL_PATH", "/lustre/fsw/portfolios/coreai/users/weijiac/models/deepseek-ai/DeepSeek-V4-Flash"
)
MAX_NEW_TOKENS = 100

PROMPTS = [
    (
        "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>What is 17 * 13?<\uff5cAssistant\uff5c></think>",
        "221",
    ),
    (
        "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>If a train travels at 60 mph and needs to cover 150 miles, how many minutes will the trip take?<\uff5cAssistant\uff5c></think>",
        "150",
    ),
    (
        "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>Write a Python function that returns the nth Fibonacci number.<\uff5cAssistant\uff5c></think>",
        "def",
    ),
]

TP = int(os.environ.get("TP_SIZE", "1"))
ETP = int(os.environ.get("ETP_SIZE", "4"))
rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))


def log(msg):
    if rank == 0:
        print(msg, flush=True)


# Phase 1: Init
log("=== Phase 1: Init ===")
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
log(f"  TP={TP} ETP={ETP}")

# Phase 2: Import
log("\n=== Phase 2: Import ===")
t0 = time.time()
from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.utils.common_utils import disable_mtp_for_inference, get_last_rank


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

# Phase 3: Generate
log("\n=== Phase 3: Generate ===")
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
stop_tokens = [tokenizer.eos_token_id]


class SingleBatchIterator:
    def __init__(self, ids, pos):
        self.batch = {"tokens": ids, "position_ids": pos}
        self._d = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._d:
            raise StopIteration
        self._d = True
        return self.batch


def fwd_step(di, model, **_):
    b = next(di)
    return model(input_ids=b["tokens"], position_ids=b["position_ids"], attention_mask=None), lambda x, **k: x


all_passed = True
for pi, (prompt, expected_substr) in enumerate(PROMPTS):
    log(f"\n--- Prompt {pi + 1}/{len(PROMPTS)} ---")
    log(f"  Q: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    generated_ids = input_ids.clone()

    for step in range(MAX_NEW_TOKENS):
        pos = torch.arange(generated_ids.size(1), device="cuda").unsqueeze(0)
        with torch.no_grad():
            out = get_forward_backward_func()(
                forward_step_func=fwd_step,
                data_iterator=SingleBatchIterator(generated_ids, pos),
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=generated_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(out, list) and out:
                out = out[0]

            if parallel_state.is_pipeline_last_stage():
                ws = parallel_state.get_tensor_model_parallel_world_size()
                if ws > 1:
                    gathered = [torch.zeros_like(out) for _ in range(ws)]
                    dist.all_gather(gathered, out, group=parallel_state.get_tensor_model_parallel_group())
                    full_logits = torch.cat(gathered, dim=2)
                else:
                    full_logits = out
                next_id = full_logits[:, -1].argmax(dim=-1, keepdim=True)

                if step < 5 and rank == 0:
                    logits = full_logits[0, -1]
                    top5_v, top5_i = logits.topk(5)
                    top5_tok = [tokenizer.decode([i]) for i in top5_i]
                    log(f"  step {step}: top5={list(zip(top5_tok, [f'{v:.2f}' for v in top5_v.tolist()]))}")
            else:
                next_id = torch.ones((1, 1), device="cuda", dtype=torch.long)

            dist.broadcast(next_id, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_id], dim=-1)

            if next_id.item() in stop_tokens:
                break

    if rank == 0:
        full_text = tokenizer.decode(generated_ids[0].tolist())
        answer = full_text[len(prompt) :]
        log(f"  A: {answer}")

        # Verify expected substring in answer
        if expected_substr in answer:
            log(f"  PASS: found '{expected_substr}' in answer")
        else:
            log(f"  FAIL: '{expected_substr}' NOT found in answer")
            all_passed = False
    dist.barrier()

if rank == 0:
    log(f"\n=== {'ALL PASSED' if all_passed else 'SOME FAILED'} ===")

dist.destroy_process_group()
