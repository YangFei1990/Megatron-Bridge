# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Smoke tests for answer_only_vocab (vocab slice) under PP, TP, and CP parallelism.

Each test must be launched in its own torchrun invocation with --nproc_per_node=2:

    CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 -m pytest \\
        tests/functional_tests/test_groups/training/test_answer_only_vocab_parallel.py::test_pp2 -sv

    CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 -m pytest \\
        tests/functional_tests/test_groups/training/test_answer_only_vocab_parallel.py::test_tp2 -sv

    CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 -m pytest \\
        tests/functional_tests/test_groups/training/test_answer_only_vocab_parallel.py::test_cp2 -sv

Running multiple tests in a single torchrun session is not supported because megatron
parallel state is not reset between pytest test functions.
"""

from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.vocab_slice import create_vocab_sliced_forward_step
from tests.functional_tests.utils import initialize_distributed


VOCAB_SIZE = 10_000
SEQ_LENGTH = 256


@dataclass
class _TinyGPTProvider(GPTModelProvider):
    """Minimal GPT provider that fits on 1 GPU and needs no pretrained weights."""

    num_layers: int = 2
    hidden_size: int = 256
    ffn_hidden_size: int = 512
    num_attention_heads: int = 4
    num_query_groups: int = 4
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    share_embeddings_and_output_weights: bool = False
    bias_activation_fusion: bool = False
    masked_softmax_fusion: bool = False
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = False
    apply_rope_fusion: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    rotary_base: int = 10_000
    layernorm_epsilon: float = 1e-5
    init_method_std: float = 0.01
    vocab_size: int | None = None


def _build_config(*, tp: int = 1, pp: int = 1, cp: int = 1) -> ConfigContainer:
    from megatron.bridge.training.mixed_precision import bf16_mixed

    return ConfigContainer(
        model=_TinyGPTProvider(
            seq_length=SEQ_LENGTH,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            sequence_parallel=(tp > 1),
        ),
        mixed_precision=bf16_mixed() if cp > 1 else None,
        train=TrainingConfig(
            train_iters=3,
            global_batch_size=4,
            micro_batch_size=1,
        ),
        validation=ValidationConfig(
            eval_interval=100,
            eval_iters=0,
        ),
        optimizer=OptimizerConfig(
            optimizer="adam",
            bf16=True,
            fp16=False,
            lr=1e-3,
            min_lr=1e-5,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            use_distributed_optimizer=False,
            clip_grad=1.0,
            weight_decay=0.01,
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="cosine",
            lr_warmup_iters=1,
            lr_warmup_init=0.0,
            lr_decay_iters=3,
            override_opt_param_scheduler=True,
            start_weight_decay=0.01,
            end_weight_decay=0.01,
            weight_decay_incr_style="constant",
        ),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=False,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            use_distributed_optimizer=False,
        ),
        dataset=MockGPTDatasetConfig(
            random_seed=42,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=SEQ_LENGTH,
            num_dataset_builder_threads=1,
            data_sharding=False,
            dataloader_type="single",
            num_workers=0,
        ),
        logger=LoggerConfig(log_interval=1),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=VOCAB_SIZE,
        ),
        checkpoint=CheckpointConfig(
            save_interval=0,
        ),
        rng=RNGConfig(seed=1234),
    )


def _make_on_train_ds_hook(install_log: list[str]):
    """Return an _on_train_ds hook that installs vocab slicing with a fixed small active set."""
    active_ids = torch.tensor(list(range(50, 150)), dtype=torch.long)  # 100 active tokens

    def on_train_ds(train_ds, fwd_step):
        install_log.append("hook_called")
        return create_vocab_sliced_forward_step(active_ids, fwd_step)

    return on_train_ds


@pytest.mark.run_only_on("GPU")
def test_pp2():
    """PP=2: last stage installs vocab slice; first stage silently skips. No crash."""
    initialize_distributed()

    install_log: list[str] = []
    hook = _make_on_train_ds_hook(install_log)

    cfg = _build_config(tp=1, pp=2, cp=1)

    try:
        pretrain(cfg, forward_step, _on_train_ds=hook)
    except Exception as e:
        pytest.fail(f"pretrain raised an exception with PP=2: {e}")

    assert "hook_called" in install_log, "on_train_ds hook was never called"


@pytest.mark.run_only_on("GPU")
def test_tp2():
    """TP=2: vocab slice is skipped with a warning; training still completes."""
    initialize_distributed()

    install_log: list[str] = []
    hook = _make_on_train_ds_hook(install_log)

    cfg = _build_config(tp=2, pp=1, cp=1)

    try:
        pretrain(cfg, forward_step, _on_train_ds=hook)
    except Exception as e:
        pytest.fail(f"pretrain raised an exception with TP=2: {e}")

    assert "hook_called" in install_log, "on_train_ds hook was never called"


@pytest.mark.run_only_on("GPU")
def test_cp2():
    """CP=2: vocab slice installs normally; context parallelism does not interfere."""
    initialize_distributed()

    install_log: list[str] = []
    hook = _make_on_train_ds_hook(install_log)

    cfg = _build_config(tp=1, pp=1, cp=2)

    try:
        pretrain(cfg, forward_step, _on_train_ds=hook)
    except Exception as e:
        pytest.fail(f"pretrain raised an exception with CP=2: {e}")

    assert "hook_called" in install_log, "on_train_ds hook was never called"
