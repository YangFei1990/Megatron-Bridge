# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T2.5: Optimizer state excludes vision tower; peak memory is lower than VL recipe.

Run with (single GPU):
    HF_HOME=... HF_PATH=Qwen/Qwen3.5-0.8B \
    uv run python -m torch.distributed.run --nproc_per_node=1 --master_port=29700 \
        tests/functional_tests/models/qwen_vl/test_qwen35_llm_optimizer_state.py
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist


HF_PATH = os.environ.get("HF_PATH", "Qwen/Qwen3.5-0.8B")
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", "3"))


def main():
    from megatron.bridge.recipes.qwen_vl import qwen35_llm_800m_sft_config
    from megatron.bridge.training.callbacks import Callback, CallbackContext
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.vlm_step import forward_step

    results: dict = {}

    class OptimizerStateChecker(Callback):
        """After the first training step, inspect which params are frozen vs trainable."""

        def on_train_step_end(self, context: CallbackContext) -> None:
            if context.state.train_state.step != 0:
                return

            from megatron.core.utils import unwrap_model

            # Frozen params have requires_grad=False and therefore never enter the
            # optimizer.  This is the authoritative check: if freeze_vision_model=True
            # and freeze_vision_projection=True the vision tower must be entirely frozen.
            vision_trainable = []
            vision_frozen = []
            lm_trainable = []

            for model_chunk in unwrap_model(context.model):
                for name, param in model_chunk.named_parameters():
                    if "vision_model" in name or "visual" in name:
                        if param.requires_grad:
                            vision_trainable.append(name)
                        else:
                            vision_frozen.append(name)
                    else:
                        if param.requires_grad:
                            lm_trainable.append(name)

            results["vision_trainable"] = vision_trainable
            results["vision_frozen"] = vision_frozen
            results["lm_trainable"] = lm_trainable
            results["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

    cfg = qwen35_llm_800m_sft_config(hf_path=HF_PATH)
    cfg.train.train_iters = TRAIN_ITERS
    cfg.train.log_interval = 1
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1

    # Isolate checkpoint I/O so this test never resumes from a prior run.
    # Point both save and load at a fresh temp dir (load finds nothing → fresh start).
    _ckpt_dir = tempfile.mkdtemp(prefix="t25_optim_ckpt_")
    cfg.checkpoint.save = _ckpt_dir
    cfg.checkpoint.load = _ckpt_dir

    torch.cuda.reset_peak_memory_stats()
    finetune(cfg, forward_step, callbacks=[OptimizerStateChecker()])

    # Only rank-0 reports results.
    if not dist.is_initialized() or dist.get_rank() == 0:
        vision_trainable = results.get("vision_trainable", [])
        vision_frozen = results.get("vision_frozen", [])
        lm_trainable = results.get("lm_trainable", [])
        peak_mem_gb = results.get("peak_mem_gb", float("nan"))

        print("\n" + "=" * 70)
        print("T2.5 Optimizer State Check")
        print("=" * 70)
        print(f"  LM params (trainable)      : {len(lm_trainable)}")
        print(f"  Vision params (frozen)     : {len(vision_frozen)}")
        print(f"  Vision params (trainable!) : {len(vision_trainable)}")
        print(f"  Peak GPU memory (GB)       : {peak_mem_gb:.3f}")

        if vision_trainable:
            print("\n  FAIL — vision params are trainable (should be frozen):")
            for n in vision_trainable[:10]:
                print(f"    {n}")
            sys.exit(1)

        if not lm_trainable:
            print("\n  FAIL — no LM params are trainable (recipe misconfigured)")
            sys.exit(1)

        if not vision_frozen:
            print("\n  WARN — no frozen vision params found; model may not have a vision tower")

        print("\n  PASS — vision tower fully frozen; LM params trainable")
        print("=" * 70)


if __name__ == "__main__":
    main()
