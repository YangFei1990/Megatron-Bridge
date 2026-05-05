#!/bin/bash
# CI_TIMEOUT=60
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

# Functional tests for the Qwen3.5 LLM-only SFT recipe.
#
# Covers:
#   T2.2  skip_megatron_param_globs excludes vision-tower weights (pytest, 1 GPU)
#   T2.5  Optimizer state excludes vision tower; peak memory check (1 GPU)
#   T2.6  LM weight parity vs HF checkpoint (1 GPU)
#   T2.7  Checkpoint save-and-resume (1 GPU)
#   T2.8  TP=2 and PP=2 smoke tests (2 GPUs each)

set -xeuo pipefail

# ---------------------------------------------------------------------------
# T2.2 — skip_megatron_param_globs vision-tower exclusion (pytest, single GPU)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="0"

uv run coverage run \
    --data-file=/opt/Megatron-Bridge/.coverage \
    --source=/opt/Megatron-Bridge/ \
    --parallel-mode \
    -m pytest \
    -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_sft_skip_vision.py

# ---------------------------------------------------------------------------
# T2.5 — Optimizer state: vision frozen, LM trainable (1 GPU)
# ---------------------------------------------------------------------------
uv run python -m torch.distributed.run \
    --nproc_per_node=1 --nnodes=1 --master_port=29700 \
    tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_optimizer_state.py

# ---------------------------------------------------------------------------
# T2.6 — LM weight parity: Megatron weights == HF source (1 GPU)
# ---------------------------------------------------------------------------
uv run python -m torch.distributed.run \
    --nproc_per_node=1 --nnodes=1 --master_port=29710 \
    tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_weight_parity.py

# ---------------------------------------------------------------------------
# T2.7 — Checkpoint save-and-resume (1 GPU)
# ---------------------------------------------------------------------------
uv run python -m torch.distributed.run \
    --nproc_per_node=1 --nnodes=1 --master_port=29711 \
    tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_resume.py

# ---------------------------------------------------------------------------
# T2.8 — TP=2 and PP=2 smoke tests (2 GPUs each)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="0,1"

MODE=tp2 uv run python -m torch.distributed.run \
    --nproc_per_node=2 --nnodes=1 --master_port=29720 \
    tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_tp_pp.py

MODE=pp2 uv run python -m torch.distributed.run \
    --nproc_per_node=2 --nnodes=1 --master_port=29721 \
    tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_tp_pp.py

coverage combine -q || true
