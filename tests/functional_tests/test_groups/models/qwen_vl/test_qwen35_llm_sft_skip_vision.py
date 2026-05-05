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

"""Functional test: skip_megatron_param_globs excludes vision-tower weights.

T2.2 from docs/verification/pr3037_skip_megatron_param_globs.md

Verifies that when the LLM-only recipe skips vision params:
  - stream_weights_hf_to_megatron yields no vision_model.* tasks
  - language_model params ARE yielded (skip is not over-eager)

Uses a real HuggingFace checkpoint (Qwen/Qwen3.5-0.8B, which is a VLM) but
mocks the Megatron module references so no distributed init is required.

Run with:
    HF_HOME=... HF_PATH=Qwen/Qwen3.5-0.8B \
    uv run python -m pytest \
        tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_sft_skip_vision.py -v
"""

import os
import unittest.mock as mock

import pytest
import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion import model_bridge as _model_bridge
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask


HF_PATH = os.environ.get("HF_PATH", "Qwen/Qwen3.5-0.8B")
SKIP_GLOBS = ["*vision_model*"]


@pytest.fixture(scope="module")
def auto_bridge():
    return AutoBridge.from_hf_pretrained(HF_PATH)


@pytest.fixture(scope="module")
def all_megatron_param_names(auto_bridge):
    """Collect all Megatron parameter names from the bridge's mapping registry."""
    inner_bridge = _model_bridge.get_model_bridge(
        auto_bridge._causal_lm_architecture,
        hf_config=auto_bridge.hf_pretrained.config,
    )
    mappings = inner_bridge.mapping_registry()
    names = []
    for m in mappings:
        mp = m.megatron_param
        if isinstance(mp, str):
            names.append(mp)
        else:
            names.extend(mp if mp else [])
    return names


@pytest.fixture(scope="module")
def stream_tasks_with_skip(auto_bridge):
    """Return (yielded_names_with_skip, yielded_names_without_skip) tuples."""
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    inner_bridge = _model_bridge.get_model_bridge(
        auto_bridge._causal_lm_architecture,
        hf_config=auto_bridge.hf_pretrained.config,
    )
    mappings = inner_bridge.mapping_registry()

    def _make_task(mp):
        name = mp if isinstance(mp, str) else list(mp)[0]
        mapping = mock.Mock()
        mapping.hf_param = "hf.dummy"
        mapping.hf_to_megatron = lambda hf_w, mod: torch.zeros(1)
        return WeightConversionTask(
            param_name=name,
            global_param_name=name,
            mapping=mapping,
            megatron_module=mock.Mock(),
            vp_stage=0,
        )

    tasks = [_make_task(m.megatron_param) for m in mappings]

    hf_pretrained = mock.Mock()
    hf_pretrained.state = {"hf.dummy": torch.zeros(1)}
    megatron_stages = [mock.MagicMock()]

    with_skip = list(
        MegatronModelBridge.stream_weights_hf_to_megatron(
            inner_bridge,
            hf_pretrained,
            megatron_stages,
            conversion_tasks=tasks,
            skip_megatron_param_globs=SKIP_GLOBS,
        )
    )
    without_skip = list(
        MegatronModelBridge.stream_weights_hf_to_megatron(
            inner_bridge,
            hf_pretrained,
            megatron_stages,
            conversion_tasks=tasks,
            skip_megatron_param_globs=None,
        )
    )
    return [t.param_name for t in with_skip], [t.param_name for t in without_skip]


def test_vision_params_excluded_by_skip_globs(all_megatron_param_names, stream_tasks_with_skip):
    """Vision-tower parameters must not appear in the stream when skip globs are set."""
    names_with_skip, _ = stream_tasks_with_skip

    vision_in_mapping = [n for n in all_megatron_param_names if "vision_model" in n]
    assert vision_in_mapping, "No vision_model params in mapping — is Qwen3.5-0.8B a VLM? Check HF_PATH."

    vision_leaked = [n for n in names_with_skip if "vision_model" in n]
    assert vision_leaked == [], f"Vision params leaked into stream despite skip globs: {vision_leaked}"


def test_language_params_present_with_skip_globs(stream_tasks_with_skip):
    """Language-model parameters must still be yielded when skip globs exclude vision only."""
    names_with_skip, _ = stream_tasks_with_skip
    lm_names = [n for n in names_with_skip if "vision_model" not in n]
    assert len(lm_names) > 0, "No language-model params in stream — skip glob is over-eager"


def test_vision_params_present_without_skip_globs(all_megatron_param_names, stream_tasks_with_skip):
    """Without skip globs, vision params must appear (validates the comparison baseline)."""
    _, names_without_skip = stream_tasks_with_skip

    vision_in_mapping = [n for n in all_megatron_param_names if "vision_model" in n]
    if not vision_in_mapping:
        pytest.skip("No vision_model tasks — model may not be a VLM")

    vision_present = [n for n in names_without_skip if "vision_model" in n]
    assert len(vision_present) > 0, "Vision params missing from stream when globs=None — baseline is invalid"
