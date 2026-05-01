"""
dsv4_tiny_ref_vs_mg.py — Compare official inference/model.py vs Megatron Bridge.

Mocks kernel.py functions with pure PyTorch, then loads the same tiny model
into both the official reference and Megatron, comparing layer by layer.

Usage: 1 GPU, TP=1.
"""

import json
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ── Step 1: Mock kernel.py before importing model.py ───────────────────────


def mock_act_quant(x, *args, **kwargs):
    return x, None


def mock_fp8_gemm(x, s, w, ws, *args):
    return F.linear(x, w)


def mock_fp4_gemm(x, s, w, ws, *args):
    return F.linear(x, w)


def mock_fp4_act_quant(x, *args, **kwargs):
    pass  # in-place no-op for BF16


def mock_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Simple dense causal attention (for short sequences, equivalent to sparse)."""
    b, sq, nh, hd = q.shape
    sk = kv.shape[1]
    # kv is [b, sk, hd] (single head), expand to [b, sk, nh, hd]
    k = kv.unsqueeze(2).expand(-1, -1, nh, -1)
    v = k  # key == value in MQA
    scores = torch.einsum("bsnh,btnh->bnst", q, k) * softmax_scale
    scores = scores + attn_sink.view(1, nh, 1, 1)
    # Causal mask
    mask = torch.triu(torch.ones(sq, sk, device=q.device, dtype=torch.bool), diagonal=1 + sk - sq)
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.einsum("bnst,btnh->bsnh", attn, v)
    return out.contiguous()


def mock_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """Pure PyTorch hc_split_sinkhorn matching the CUDA kernel."""
    # mixes: [b, s, (2+hc)*hc] — already RMS-scaled projection
    # hc_scale: [3] — [alpha_pre, alpha_post, alpha_res]
    # hc_base: [(2+hc)*hc] — bias
    hc = hc_mult

    # Apply scale and base: h = mixes * alpha + base
    alpha_pre, alpha_post, alpha_res = hc_scale[0], hc_scale[1], hc_scale[2]
    alpha = torch.cat(
        [
            alpha_pre.expand(hc),
            alpha_post.expand(hc),
            alpha_res.expand(hc * hc),
        ]
    )
    h = mixes * alpha + hc_base

    pre = torch.sigmoid(h[..., :hc])
    post = h[..., hc : 2 * hc]  # raw (no activation in reference)
    comb_logits = h[..., 2 * hc :]

    # Log-domain Sinkhorn
    b_sz = comb_logits.shape[:-1]
    comb = comb_logits.reshape(*b_sz, hc, hc)
    row_max = comb.max(dim=-1, keepdim=True).values
    M = torch.exp(comb - row_max)
    for _ in range(sinkhorn_iters):
        M = M / M.sum(-1, keepdim=True).clamp(min=eps)
        M = M / M.sum(-2, keepdim=True).clamp(min=eps)

    return pre, post, M


# Install mocks as a fake "kernel" module
import types


kernel_mock = types.ModuleType("kernel")
kernel_mock.act_quant = mock_act_quant
kernel_mock.fp4_act_quant = mock_fp4_act_quant
kernel_mock.fp8_gemm = mock_fp8_gemm
kernel_mock.fp4_gemm = mock_fp4_gemm
kernel_mock.sparse_attn = mock_sparse_attn
kernel_mock.hc_split_sinkhorn = mock_hc_split_sinkhorn
sys.modules["kernel"] = kernel_mock


# Also mock fast_hadamard_transform (model.py imports it directly in rotate_activation)
def _pytorch_hadamard(x, scale=1.0):
    n = x.shape[-1]
    result = x.clone()
    h = 1
    while h < n:
        result = result.view(*result.shape[:-1], -1, 2 * h)
        a = result[..., :h].clone()
        b = result[..., h:].clone()
        result[..., :h] = a + b
        result[..., h:] = a - b
        result = result.view(*result.shape[:-2], -1)
        h *= 2
    return result * scale


fht_mock = types.ModuleType("fast_hadamard_transform")
fht_mock.hadamard_transform = _pytorch_hadamard
import importlib.machinery


fht_mock.__spec__ = importlib.machinery.ModuleSpec("fast_hadamard_transform", None)
sys.modules["fast_hadamard_transform"] = fht_mock


# ── Step 2: Tiny config ───────────────────────────────────────────────────

# First: test with all compress_ratio=0 to validate HC + attention + MoE.
# Indexer (compress_ratio=4/128) has a dtype mismatch bug — test separately later.
TINY_CONFIG = {
    "model_type": "deepseek_v4",
    "hidden_size": 256,
    "num_hidden_layers": 6,
    "num_hash_layers": 3,
    "compress_ratios": [0, 0, 0, 0, 0, 0],
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "q_lora_rank": 128,
    "o_lora_rank": 128,
    "qk_rope_head_dim": 16,
    "o_groups": 2,
    "index_n_heads": 8,
    "index_head_dim": 16,
    "index_topk": 32,
    "sliding_window": 16,
    "moe_intermediate_size": 256,
    "n_routed_experts": 8,
    "num_experts_per_tok": 1,  # topk=1 avoids TE grouped_linear dispatch bug with topk>1
    "n_shared_experts": 1,
    "hc_mult": 4,
    "hc_eps": 1e-6,
    "hc_sinkhorn_iters": 20,
    "vocab_size": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "compress_rope_theta": 160000.0,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "swiglu_limit": 10.0,
}

# Use 32 tokens to avoid TE grouped_linear dispatch bug with very small batches
INPUT_IDS = list(range(10, 42))  # [10, 11, 12, ..., 41] — 32 tokens


def log(msg):
    print(msg, flush=True)


# ── Step 3: Create reference model (inference/model.py) ───────────────────


def create_ref_model():
    """Import and instantiate the official inference/model.py with tiny config."""
    # Try OCI path first, fall back to CW-DFW path
    model_dir = None
    for p in [
        "/lustre/fsw/portfolios/coreai/users/weijiac/models/deepseek-ai/DeepSeek-V4-Flash/inference",
        "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/nemo_home/models/deepseek/DeepSeek-V4-Flash/inference",
    ]:
        if os.path.exists(os.path.join(p, "model.py")):
            model_dir = p
            break
    assert model_dir is not None, "Cannot find inference/model.py on any known path"
    sys.path.insert(0, model_dir)

    import model as ref_model

    # Override globals for single-GPU BF16
    ref_model.world_size = 1
    ref_model.rank = 0
    ref_model.default_dtype = torch.bfloat16

    cfg = TINY_CONFIG
    args = ref_model.ModelArgs(
        max_batch_size=1,
        max_seq_len=64,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=cfg["vocab_size"],
        dim=cfg["hidden_size"],
        moe_inter_dim=cfg["moe_intermediate_size"],
        n_layers=cfg["num_hidden_layers"],
        n_hash_layers=cfg["num_hash_layers"],
        n_mtp_layers=0,
        n_heads=cfg["num_attention_heads"],
        n_routed_experts=cfg["n_routed_experts"],
        n_shared_experts=cfg["n_shared_experts"],
        n_activated_experts=cfg["num_experts_per_tok"],
        score_func=cfg["scoring_func"],
        route_scale=cfg["routed_scaling_factor"],
        swiglu_limit=cfg["swiglu_limit"],
        q_lora_rank=cfg["q_lora_rank"],
        head_dim=cfg["head_dim"],
        rope_head_dim=cfg["qk_rope_head_dim"],
        norm_eps=cfg["rms_norm_eps"],
        o_groups=cfg["o_groups"],
        o_lora_rank=cfg["o_lora_rank"],
        window_size=cfg["sliding_window"],
        compress_ratios=tuple(cfg["compress_ratios"][: cfg["num_hidden_layers"]]),
        compress_rope_theta=cfg["compress_rope_theta"],
        rope_theta=cfg["rope_theta"],
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=cfg["index_n_heads"],
        index_head_dim=cfg["index_head_dim"],
        index_topk=cfg["index_topk"],
        hc_mult=cfg["hc_mult"],
        hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"],
        hc_eps=cfg["hc_eps"],
    )

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    m = ref_model.Transformer(args)
    m.eval()
    return m, ref_model


# ── Step 4: Run reference forward with hooks ──────────────────────────────


def run_ref_forward(ref_m, input_ids):
    layer_outs = {}
    hooks = []
    for i, layer in enumerate(ref_m.layers):

        def make_hook(idx):
            def fn(mod, inp, out):
                if isinstance(out, torch.Tensor):
                    layer_outs[idx] = out.detach().float().cpu()

            return fn

        hooks.append(layer.register_forward_hook(make_hook(i)))

    ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        logits = ref_m(ids, start_pos=0)
    for h in hooks:
        h.remove()
    return logits.float().cpu(), layer_outs


# ── Step 5: Create Megatron model and import same weights ─────────────────


def init_distributed():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    for k, v in [("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")]:
        os.environ.setdefault(k, v)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", world_size=1, rank=0)
    from megatron.core import parallel_state, tensor_parallel

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(42)


def create_megatron_model(config_dir):
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
    from megatron.bridge.utils.common_utils import disable_mtp_for_inference

    bridge = AutoBridge.from_hf_pretrained(config_dir)
    hf = PreTrainedCausalLM(config_dir)
    prov = bridge._model_bridge.provider_bridge(hf)
    prov.tensor_model_parallel_size = 1
    prov.pipeline_model_parallel_size = 1
    prov.expert_model_parallel_size = 1
    prov.expert_tensor_parallel_size = 1
    prov.finalize()
    model = prov.provide_distributed_model(wrap_with_ddp=False)
    for m in model:
        m.eval()
        disable_mtp_for_inference(m)
    return model, bridge


def run_megatron_forward(mg_model, input_ids):
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    mg = mg_model[0]
    if hasattr(mg, "module"):
        mg = mg.module
    if hasattr(mg, "language_model"):
        mg = mg.language_model
    decoder = getattr(mg, "decoder", None)

    layer_outs = {}
    hooks = []
    if decoder and hasattr(decoder, "layers"):
        for i, layer in enumerate(decoder.layers):

            def make_hook(idx):
                def fn(mod, inp, out):
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if isinstance(t, torch.Tensor):
                        layer_outs[idx] = t.detach().float().cpu()

                return fn

            hooks.append(layer.register_forward_hook(make_hook(i)))

    ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    pos = torch.arange(len(input_ids), device="cuda").unsqueeze(0)

    class It:
        def __init__(self):
            self._d = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._d:
                raise StopIteration
            self._d = True
            return {"tokens": ids, "position_ids": pos}

    def fwd(di, model, **_):
        b = next(di)
        return model(input_ids=b["tokens"], position_ids=b["position_ids"], attention_mask=None), lambda x, **k: x

    f = get_forward_backward_func()
    with torch.no_grad():
        out = f(
            forward_step_func=fwd,
            data_iterator=It(),
            model=mg_model,
            num_microbatches=1,
            forward_only=True,
            seq_length=len(input_ids),
            micro_batch_size=1,
            collect_non_loss_data=True,
        )
    for h in hooks:
        h.remove()
    logits = out[0].detach().float().cpu() if isinstance(out, list) and out else None
    return logits, layer_outs


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    log("=== DSv4 Tiny: Official Reference vs Megatron ===\n")

    init_distributed()

    # Create reference model
    log("Creating reference model (inference/model.py)...")
    ref_m, ref_mod = create_ref_model()
    log(f"  {sum(p.numel() for p in ref_m.parameters())} params")

    # Save reference weights as HF-format checkpoint for Bridge import
    # (We need to map ref param names → HF checkpoint names)
    log("\nSaving reference weights as HF checkpoint...")
    tmpdir = tempfile.mkdtemp(prefix="dsv4_ref_")

    # Save config.json
    hf_config = dict(TINY_CONFIG)
    hf_config["architectures"] = ["DeepseekV4ForCausalLM"]
    hf_config["num_nextn_predict_layers"] = 0
    hf_config["attention_bias"] = False
    hf_config["attention_dropout"] = 0.0
    hf_config["hidden_act"] = "silu"
    hf_config["initializer_range"] = 0.02
    hf_config["tie_word_embeddings"] = False
    hf_config["torch_dtype"] = "bfloat16"
    hf_config["max_position_embeddings"] = 256
    hf_config["norm_topk_prob"] = True
    hf_config["topk_method"] = "noaux_tc"
    hf_config["rope_scaling"] = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 256,
        "type": "yarn",
    }
    # Ensure compress_ratios in config matches num_hidden_layers (no MTP entry)
    hf_config["compress_ratios"] = TINY_CONFIG["compress_ratios"][: TINY_CONFIG["num_hidden_layers"]]
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    # Save weights in DeepSeek checkpoint naming (what the bridge expects)
    from safetensors.torch import save_file

    ref_sd = {
        k: v.cpu().contiguous()
        for k, v in ref_m.state_dict().items()
        if not k.startswith("freqs_cis") and "kv_cache" not in k
    }
    save_file(ref_sd, os.path.join(tmpdir, "model.safetensors"))
    log(f"  Saved {len(ref_sd)} tensors to {tmpdir}")

    # Reference forward
    log("\nReference forward...")
    ref_logits, ref_layers = run_ref_forward(ref_m, INPUT_IDS)
    log(f"  {len(ref_layers)} layers, logits: {list(ref_logits.shape)}")

    # Free reference model
    del ref_m
    torch.cuda.empty_cache()

    # Create Megatron model (import same weights via Bridge)
    log("\nCreating Megatron model (Bridge import)...")
    mg_model, bridge = create_megatron_model(tmpdir)
    bridge.load_hf_weights(mg_model)
    log("  Done")

    # Megatron forward
    log("\nMegatron forward...")
    mg_logits, mg_layers = run_megatron_forward(mg_model, INPUT_IDS)
    log(f"  {len(mg_layers)} layers, logits: {list(mg_logits.shape) if mg_logits is not None else 'None'}")

    # Compare
    log("\n" + "=" * 70)
    log("Layer-by-layer: Reference (model.py) vs Megatron (same weights)")
    log("=" * 70)
    for i in sorted(ref_layers.keys()):
        if i not in mg_layers:
            log(f"  Layer {i}: MISSING in Megatron")
            continue
        ref_t = ref_layers[i].flatten()
        mg_t = mg_layers[i].flatten()
        if ref_t.shape != mg_t.shape:
            log(f"  Layer {i}: SHAPE MISMATCH ref={list(ref_layers[i].shape)} mg={list(mg_layers[i].shape)}")
            log(f"           ref_norm={ref_t.norm():.4f} mg_norm={mg_t.norm():.4f}")
            continue
        cos = F.cosine_similarity(ref_t.unsqueeze(0), mg_t.unsqueeze(0)).item()
        md = (ref_t - mg_t).abs().max().item()
        log(
            f"  Layer {i}: cos_sim={cos:.8f}  max_diff={md:.2e}  ref_norm={ref_t.norm():.4f}  mg_norm={mg_t.norm():.4f}"
        )

    if mg_logits is not None:
        ref_l = ref_logits[0, -1] if ref_logits.dim() == 3 else ref_logits[-1]
        mg_l = mg_logits[0, -1] if mg_logits.dim() == 3 else mg_logits[-1]
        v = min(ref_l.shape[0], mg_l.shape[0])
        cos = F.cosine_similarity(ref_l[:v].unsqueeze(0), mg_l[:v].unsqueeze(0)).item()
        md = (ref_l[:v] - mg_l[:v]).abs().max().item()
        log(f"\n  Logits: cos_sim={cos:.8f}  max_diff={md:.2e}")
        log(f"  Ref top-5: {ref_l.topk(5).indices.tolist()}")
        log(f"  MG  top-5: {mg_l[:v].topk(5).indices.tolist()}")

    log("\nDone.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
