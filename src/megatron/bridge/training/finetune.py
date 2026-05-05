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

from megatron.bridge.training.callbacks import Callback, CallbackManager
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.utils.common_utils import print_rank_0
from megatron.bridge.utils.decorators import experimental_fn


@experimental_fn
def finetune(
    config: ConfigContainer,
    forward_step_func: ForwardStepCallable,
    callbacks: list[Callback] | CallbackManager | None = None,
) -> None:
    """Main function to run the finetuning.

    Args:
        config: The main configuration container holding all necessary parameters.
        forward_step_func: A callable (function or functor) that performs a single
                          forward and backward step, returning the loss and any computed
                          metrics. Supports the following signatures:
                          - 2 args: (data_iterator, model)
                          - 3 args: (data_iterator, model, return_schedule_plan=False)
                                   OR (state: GlobalState, data_iterator, model)
                          - 4 args: (state: GlobalState, data_iterator, model, return_schedule_plan=False)
        callbacks: Optional list of Callback instances, a CallbackManager, or None.

    Note:
        Use the signature with GlobalState type hint for full access to configuration, timers, and training state.
        State injection is automatic based on type hints or parameter names.
        Functors (classes with __call__) are fully supported.

    Warnings:
        This is an experimental API and is subject to change in backwards
        incompatible ways without notice.
    """
    assert config.checkpoint.pretrained_checkpoint is not None or config.checkpoint.load is not None, (
        "Finetuning requires a loading from a pretrained checkpoint or resuming from a checkpoint"
    )

    on_train_ds = None
    if isinstance(config.dataset, FinetuningDatasetConfig) and config.dataset.answer_only_vocab:
        from megatron.bridge.training.vocab_slice import (
            collect_active_vocab_ids,
            create_vocab_sliced_forward_step,
        )

        max_samples = config.dataset.answer_only_vocab_max_samples

        def on_train_ds(train_ds, fwd_step):
            active_ids = collect_active_vocab_ids(train_ds, max_samples=max_samples)
            print_rank_0(f"[answer_only_vocab] {len(active_ids)} answer vocab IDs collected from training dataset.")
            return create_vocab_sliced_forward_step(active_ids, fwd_step)

    return pretrain(config, forward_step_func, callbacks=callbacks, _on_train_ds=on_train_ds)
