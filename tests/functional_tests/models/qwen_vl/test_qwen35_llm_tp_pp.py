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

"""T2.8: TP and PP smoke tests for the Qwen3.5 LLM-only recipe.

Runs 2 training steps under non-default parallelism settings and asserts
finite, decreasing loss.

Modes (set via MODE env var):
  tp2  — TP=2, PP=1  (2 GPUs)
  pp2  — TP=1, PP=2  (2 GPUs)

Run with:
    # TP=2 on GPUs 0,1
    MODE=tp2 HF_HOME=... HF_PATH=Qwen/Qwen3.5-0.8B \
    CUDA_VISIBLE_DEVICES=0,1 \
    uv run python -m torch.distributed.run --nproc_per_node=2 --master_port=29720 \
        tests/functional_tests/models/qwen_vl/test_qwen35_llm_tp_pp.py

    # PP=2 on GPUs 2,3
    MODE=pp2 HF_HOME=... HF_PATH=Qwen/Qwen3.5-0.8B \
    CUDA_VISIBLE_DEVICES=2,3 \
    uv run python -m torch.distributed.run --nproc_per_node=2 --master_port=29721 \
        tests/functional_tests/models/qwen_vl/test_qwen35_llm_tp_pp.py
"""

import os
import sys
import tempfile

import torch


HF_PATH = os.environ.get("HF_PATH", "Qwen/Qwen3.5-0.8B")
MODE = os.environ.get("MODE", "tp2")
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", "2"))


def main():
    from megatron.bridge.recipes.qwen_vl import qwen35_llm_800m_sft_config
    from megatron.bridge.training.callbacks import Callback, CallbackContext
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.vlm_step import forward_step

    if MODE not in ("tp2", "pp2"):
        print(f"Unknown MODE={MODE!r}. Use tp2 or pp2.", file=sys.stderr)
        sys.exit(1)

    losses: list[float] = []
    is_reporting_rank: list[bool] = [False]

    class LossCapture(Callback):
        def on_train_start(self, context: CallbackContext) -> None:
            from megatron.core import parallel_state as mpu

            # Capture whether this rank is responsible for reporting losses.
            # In PP, only the last pipeline stage has the loss; in TP only rank 0
            # within the TP group reports to avoid duplicates.
            is_last_pp = mpu.is_pipeline_last_stage()
            is_first_tp = mpu.get_tensor_model_parallel_rank() == 0
            is_reporting_rank[0] = is_last_pp and is_first_tp

        def on_train_step_end(self, context: CallbackContext) -> None:
            if not is_reporting_rank[0]:
                return
            loss_dict = context.loss_dict or {}
            loss = loss_dict.get("lm loss", float("nan"))
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            losses.append(loss)

    cfg = qwen35_llm_800m_sft_config(hf_path=HF_PATH)
    cfg.train.train_iters = TRAIN_ITERS
    cfg.train.log_interval = 1

    if MODE == "tp2":
        cfg.model.tensor_model_parallel_size = 2
        cfg.model.pipeline_model_parallel_size = 1
        cfg.model.sequence_parallel = True
    else:  # pp2
        cfg.model.tensor_model_parallel_size = 1
        cfg.model.pipeline_model_parallel_size = 2
        cfg.model.virtual_pipeline_model_parallel_size = None

    _ckpt_dir = tempfile.mkdtemp(prefix=f"t28_{MODE}_ckpt_")
    cfg.checkpoint.save = _ckpt_dir
    cfg.checkpoint.load = _ckpt_dir

    finetune(cfg, forward_step, callbacks=[LossCapture()])

    # `is_reporting_rank` was set inside on_train_start; use it now that
    # parallel_state has been torn down.
    if is_reporting_rank[0]:
        print("\n" + "=" * 70)
        print(f"T2.8 {MODE.upper()} Smoke Test")
        print("=" * 70)
        print(f"  Losses: {[f'{v:.4f}' for v in losses]}")

        failed = []
        if len(losses) < TRAIN_ITERS:
            failed.append(f"Expected {TRAIN_ITERS} loss values, got {len(losses)}")

        nan_losses = [v for v in losses if not (v == v) or v == float("inf")]
        if nan_losses:
            failed.append(f"Non-finite losses: {nan_losses}")
        else:
            print("  PASS  All losses finite")

        if failed:
            print("\n  FAIL:")
            for msg in failed:
                print(f"    {msg}")
            sys.exit(1)

        print(f"\n  PASS — {MODE.upper()} training ran {TRAIN_ITERS} steps with finite losses")
        print("=" * 70)


if __name__ == "__main__":
    main()
