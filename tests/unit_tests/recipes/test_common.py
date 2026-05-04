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

#
# Test purpose:
# - Cover the 5 base config builders in `megatron.bridge.recipes.common` —
#   `_pretrain_common`, `_sft_common`, `_peft_common`, `_sft_common_vlm`,
#   `_peft_common_vlm`. These functions are the foundation of every recipe
#   in the project but had zero direct test coverage.
# - All 5 are pure config builders with no HF I/O, so tests are fast and
#   require no mocking.
#

from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.common import (
    _peft_common,
    _peft_common_vlm,
    _pretrain_common,
    _sft_common,
    _sft_common_vlm,
)
from megatron.bridge.training.config import ConfigContainer


def _assert_basic_shape(cfg: ConfigContainer) -> None:
    """Every common config must populate these top-level fields."""
    assert isinstance(cfg, ConfigContainer)
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None
    assert cfg.dist is not None


# -----------------------------------------------------------------------------
# _pretrain_common
# -----------------------------------------------------------------------------


class TestPretrainCommon:
    """Tests for `_pretrain_common`."""

    def test_returns_valid_container_with_no_model(self):
        """Caller must set cfg.model — the helper leaves it None."""
        cfg = _pretrain_common()
        _assert_basic_shape(cfg)
        # Model is intentionally None — recipe authors set it.
        assert cfg.model is None

    def test_pretrain_training_defaults(self):
        cfg = _pretrain_common()
        assert cfg.train.train_iters == 300000
        assert cfg.train.global_batch_size == 32
        assert cfg.train.micro_batch_size == 2
        assert cfg.train.manual_gc is True
        assert cfg.train.manual_gc_interval == 100

    def test_pretrain_validation_defaults(self):
        cfg = _pretrain_common()
        assert cfg.validation.eval_interval == 500
        assert cfg.validation.eval_iters == 32

    def test_pretrain_ddp_defaults_overlap_and_dist_optimizer(self):
        cfg = _pretrain_common()
        assert cfg.ddp.overlap_grad_reduce is True
        assert cfg.ddp.overlap_param_gather is True
        assert cfg.ddp.use_distributed_optimizer is True
        assert cfg.ddp.average_in_collective is True
        assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
        assert cfg.ddp.check_for_nan_in_grad is True

    def test_pretrain_uses_huggingface_tokenizer_placeholder(self):
        cfg = _pretrain_common()
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        # tokenizer_model is a placeholder — recipe must override.
        assert cfg.tokenizer.tokenizer_model is None

    def test_pretrain_dataset_uses_mock_blend(self):
        cfg = _pretrain_common()
        assert cfg.dataset.blend is None
        assert cfg.dataset.blend_per_split is None
        assert cfg.dataset.split == "9999,8,2"
        assert cfg.dataset.seq_length == 4096
        assert cfg.dataset.skip_getting_attention_mask_from_dataset is True

    def test_pretrain_rng_seed_is_pretrain_default(self):
        """Pretrain uses seed=1234 (different from SFT/PEFT seed=5678)."""
        cfg = _pretrain_common()
        assert cfg.rng.seed == 1234

    def test_pretrain_no_peft_attached(self):
        cfg = _pretrain_common()
        # ConfigContainer's `peft` attr defaults to None for pretrain.
        assert getattr(cfg, "peft", None) is None

    def test_pretrain_mixed_precision_default(self):
        cfg = _pretrain_common()
        assert cfg.mixed_precision == "bf16_mixed"

    def test_pretrain_logger_interval(self):
        cfg = _pretrain_common()
        assert cfg.logger.log_interval == 10
        assert cfg.logger.log_timers_to_tensorboard is True


# -----------------------------------------------------------------------------
# _sft_common
# -----------------------------------------------------------------------------


class TestSftCommon:
    """Tests for `_sft_common`."""

    def test_returns_valid_container_with_no_model(self):
        cfg = _sft_common()
        _assert_basic_shape(cfg)
        assert cfg.model is None

    def test_sft_training_is_shorter_than_pretrain(self):
        cfg = _sft_common()
        # SFT defaults are smaller than pretrain's 300000.
        assert cfg.train.train_iters == 1000
        assert cfg.train.global_batch_size == 128
        assert cfg.train.micro_batch_size == 1

    def test_sft_optimizer_uses_low_lr(self):
        """Full SFT picks a lower max_lr (5e-6) than pretrain."""
        cfg = _sft_common()
        # SchedulerConfig stores lr_warmup_iters; OptimizerConfig stores `lr`.
        assert cfg.optimizer.lr == 5e-6
        # adam_beta2 is bumped to 0.98 for fine-tuning per docstring.
        assert cfg.optimizer.adam_beta2 == 0.98

    def test_sft_ddp_minimal_settings(self):
        """SFT keeps DDP settings minimal — no overlap by default."""
        cfg = _sft_common()
        assert cfg.ddp.check_for_nan_in_grad is True
        assert cfg.ddp.grad_reduce_in_fp32 is True
        # The minimal SFT DDP does NOT enable overlap by default
        # (recipes may override to True for specific models).
        assert cfg.ddp.overlap_grad_reduce is False
        assert cfg.ddp.overlap_param_gather is False

    def test_sft_rng_uses_finetune_seed(self):
        """SFT uses seed=5678 (different from pretrain seed=1234)."""
        cfg = _sft_common()
        assert cfg.rng.seed == 5678

    def test_sft_no_peft_attached(self):
        cfg = _sft_common()
        assert cfg.peft is None

    def test_sft_checkpoint_supports_pretrained_load(self):
        cfg = _sft_common()
        # The pretrained_checkpoint slot exists for full-SFT loading.
        assert cfg.checkpoint.pretrained_checkpoint is None
        assert cfg.checkpoint.save_interval == 100


# -----------------------------------------------------------------------------
# _peft_common
# -----------------------------------------------------------------------------


class TestPeftCommon:
    """Tests for `_peft_common`."""

    def test_returns_valid_container_with_no_model(self):
        cfg = _peft_common()
        _assert_basic_shape(cfg)
        assert cfg.model is None

    def test_peft_optimizer_uses_higher_lr_than_sft(self):
        """PEFT picks max_lr=1e-4 vs SFT's 5e-6."""
        cfg = _peft_common()
        assert cfg.optimizer.lr == 1e-4
        assert cfg.optimizer.adam_beta2 == 0.98

    def test_peft_attaches_default_lora(self):
        """LoRA is attached with the documented standard targets and dims."""
        cfg = _peft_common()
        assert isinstance(cfg.peft, LoRA)
        assert cfg.peft.dim == 32
        assert cfg.peft.alpha == 32
        assert cfg.peft.dropout == 0.0
        assert cfg.peft.target_modules == [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
        ]

    def test_peft_rng_uses_finetune_seed(self):
        """PEFT shares the SFT seed (5678) — both are fine-tuning paths."""
        cfg = _peft_common()
        assert cfg.rng.seed == 5678

    def test_peft_training_defaults_match_sft(self):
        """PEFT uses the same compact training cadence as SFT."""
        cfg = _peft_common()
        assert cfg.train.train_iters == 1000
        assert cfg.train.global_batch_size == 128
        assert cfg.train.micro_batch_size == 1


# -----------------------------------------------------------------------------
# _sft_common_vlm  (inherits from _sft_common, then overrides)
# -----------------------------------------------------------------------------


class TestSftCommonVlm:
    """Tests for `_sft_common_vlm`."""

    def test_returns_valid_container(self):
        cfg = _sft_common_vlm()
        _assert_basic_shape(cfg)

    def test_vlm_sft_overrides_training_defaults(self):
        """VLM SFT overrides the LLM SFT cadence: longer training, smaller GBS."""
        cfg = _sft_common_vlm()
        assert cfg.train.train_iters == 300000
        assert cfg.train.global_batch_size == 32
        assert cfg.train.micro_batch_size == 2

    def test_vlm_sft_uses_null_tokenizer(self):
        """VLMs use NullTokenizer because the processor handles tokenization."""
        cfg = _sft_common_vlm()
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
        # Vocab size is set to the project default.
        assert cfg.tokenizer.vocab_size > 0

    def test_vlm_sft_dataset_is_hf_conversation_provider(self):
        from megatron.bridge.data.vlm_datasets.hf_provider import (
            HFDatasetConversationProvider,
        )

        cfg = _sft_common_vlm()
        assert isinstance(cfg.dataset, HFDatasetConversationProvider)
        assert cfg.dataset.maker_name == "make_cord_v2_dataset"
        assert cfg.dataset.seq_length == 4096
        # hf_processor_path is a placeholder — recipe must set.
        assert cfg.dataset.hf_processor_path is None

    def test_vlm_sft_ddp_disables_overlap(self):
        """Per docstring, VLM SFT runs without grad/param overlap."""
        cfg = _sft_common_vlm()
        assert cfg.ddp.overlap_grad_reduce is False
        assert cfg.ddp.overlap_param_gather is False
        # But still uses distributed optimizer.
        assert cfg.ddp.use_distributed_optimizer is True

    def test_vlm_sft_uses_pretrain_seed(self):
        """Per docstring, VLM SFT uses RNG seed 1234 (not 5678 like LLM SFT)."""
        cfg = _sft_common_vlm()
        assert cfg.rng.seed == 1234

    def test_vlm_sft_optimizer_uses_higher_lr(self):
        """VLM SFT bumps lr to 3e-4 (vs LLM SFT 5e-6)."""
        cfg = _sft_common_vlm()
        assert cfg.optimizer.lr == 3e-4


# -----------------------------------------------------------------------------
# _peft_common_vlm  (inherits from _peft_common, then overrides)
# -----------------------------------------------------------------------------


class TestPeftCommonVlm:
    """Tests for `_peft_common_vlm`."""

    def test_returns_valid_container(self):
        cfg = _peft_common_vlm()
        _assert_basic_shape(cfg)

    def test_vlm_peft_inherits_lora_from_peft_common(self):
        """The VLM PEFT helper inherits the LoRA config from `_peft_common`."""
        cfg = _peft_common_vlm()
        assert isinstance(cfg.peft, LoRA)
        assert cfg.peft.dim == 32
        assert cfg.peft.alpha == 32

    def test_vlm_peft_uses_null_tokenizer(self):
        cfg = _peft_common_vlm()
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    def test_vlm_peft_dataset_is_hf_conversation_provider(self):
        from megatron.bridge.data.vlm_datasets.hf_provider import (
            HFDatasetConversationProvider,
        )

        cfg = _peft_common_vlm()
        assert isinstance(cfg.dataset, HFDatasetConversationProvider)
        assert cfg.dataset.hf_processor_path is None

    def test_vlm_peft_uses_pretrain_seed(self):
        """Per docstring, VLM PEFT uses RNG seed 1234."""
        cfg = _peft_common_vlm()
        assert cfg.rng.seed == 1234
