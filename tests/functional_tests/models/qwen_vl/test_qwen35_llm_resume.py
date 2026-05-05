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

"""T2.7: Checkpoint save and resume for the Qwen3.5 LLM-only recipe.

Two-phase test:
  Phase 1  Run 3 steps, save checkpoint.
  Phase 2  Resume from that checkpoint and run 2 more steps.

Assertions:
  - After resume the training loop starts at step 3 (not 0).
  - All losses are finite throughout both phases.

Run with (single GPU):
    HF_HOME=... HF_PATH=Qwen/Qwen3.5-0.8B \
    uv run python -m torch.distributed.run --nproc_per_node=1 --master_port=29711 \
        tests/functional_tests/models/qwen_vl/test_qwen35_llm_resume.py
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist


HF_PATH = os.environ.get("HF_PATH", "Qwen/Qwen3.5-0.8B")
PHASE1_ITERS = int(os.environ.get("PHASE1_ITERS", "3"))
PHASE2_ITERS = int(os.environ.get("PHASE2_ITERS", "2"))


def _make_cfg(hf_path: str, ckpt_dir: str, train_iters: int):
    from megatron.bridge.recipes.qwen_vl import qwen35_llm_800m_sft_config

    cfg = qwen35_llm_800m_sft_config(hf_path=hf_path)
    cfg.train.train_iters = train_iters
    cfg.train.log_interval = 1
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.checkpoint.save = ckpt_dir
    cfg.checkpoint.load = ckpt_dir
    cfg.checkpoint.save_interval = PHASE1_ITERS
    return cfg


def main():
    from megatron.bridge.training.callbacks import Callback, CallbackContext
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.vlm_step import forward_step

    ckpt_dir = tempfile.mkdtemp(prefix="t27_resume_ckpt_")
    phase1_losses: list[float] = []
    phase2_steps: list[int] = []
    phase2_losses: list[float] = []

    class LossRecorder(Callback):
        def __init__(self, steps_out: list, losses_out: list):
            self._steps = steps_out
            self._losses = losses_out

        def on_train_step_end(self, context: CallbackContext) -> None:
            step = context.state.train_state.step
            loss_dict = context.loss_dict or {}
            loss = loss_dict.get("lm loss", float("nan"))
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self._steps.append(step)
            self._losses.append(loss)

    # Phase 1: fresh start
    cfg1 = _make_cfg(HF_PATH, ckpt_dir, PHASE1_ITERS)
    phase1_steps: list[int] = []
    finetune(cfg1, forward_step, callbacks=[LossRecorder(phase1_steps, phase1_losses)])

    # Phase 2: resume (train_iters extended by PHASE2_ITERS)
    cfg2 = _make_cfg(HF_PATH, ckpt_dir, PHASE1_ITERS + PHASE2_ITERS)
    finetune(cfg2, forward_step, callbacks=[LossRecorder(phase2_steps, phase2_losses)])

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\n" + "=" * 70)
        print("T2.7 Checkpoint Resume Check")
        print("=" * 70)
        print(f"  Phase 1 steps : {phase1_steps}")
        print(f"  Phase 1 losses: {[f'{v:.4f}' for v in phase1_losses]}")
        print(f"  Phase 2 steps : {phase2_steps}")
        print(f"  Phase 2 losses: {[f'{v:.4f}' for v in phase2_losses]}")

        failed = []

        # Phase 2 must start at step PHASE1_ITERS (not 0)
        if not phase2_steps:
            failed.append("Phase 2 recorded no steps — training did not run after resume")
        elif phase2_steps[0] != PHASE1_ITERS:
            failed.append(
                f"Phase 2 started at step {phase2_steps[0]}, expected {PHASE1_ITERS} (checkpoint not loaded properly)"
            )
        else:
            print(f"\n  PASS  Resume starts at step {phase2_steps[0]} (not 0)")

        # All losses must be finite
        all_losses = phase1_losses + phase2_losses
        nan_losses = [v for v in all_losses if not (v == v) or v == float("inf")]
        if nan_losses:
            failed.append(f"Non-finite losses detected: {nan_losses}")
        else:
            print(f"  PASS  All {len(all_losses)} losses are finite")

        if failed:
            print("\n  FAIL:")
            for msg in failed:
                print(f"    {msg}")
            sys.exit(1)

        print("\n  PASS — checkpoint resume works correctly")
        print("=" * 70)


if __name__ == "__main__":
    main()
