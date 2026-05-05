# Answer-Only Vocabulary Slicing

When fine-tuning with `answer_only_loss`, the model only needs to predict tokens that appear
in training answers. However, the output projection and cross-entropy loss are still computed
over the full vocabulary by default. **Answer-only vocab slicing** detects which token IDs
appear in answers, restricts the output projection to those tokens, and reduces compute
proportionally.

## When to Use

Enable this feature when **both** conditions hold:

1. You are training with `answer_only_loss = True`.
2. The answer vocabulary is small relative to the full model vocabulary — typically **≤ 5 %**.

Common examples where the answer vocab is small:

- Classification tasks (the model outputs a label word such as `yes`, `no`, `A`, `B`, `C`).
- Code generation benchmarks with a narrow target language vocabulary.
- Instruction-following tasks where responses are constrained to a fixed template.

For general-purpose SFT where answers contain a wide range of tokens, the answer vocabulary
will be nearly the same size as the full vocabulary and the feature will provide little benefit.

## How It Works

1. Before training starts, `collect_active_vocab_ids` scans the training dataset and records
   every token ID that appears at an answer position (`loss_mask > 0`). This produces a sorted
   tensor of *N* unique token IDs.
2. `install_vocab_slice` monkey-patches the model:
   - **Output projection**: slices the weight matrix from `[V, H]` to `[N, H]`, where
     `V` is the full vocab size and `N` is the number of active tokens. The output is
     `[batch, seq, N]` logits instead of `[batch, seq, V]`.
   - **Loss function**: remaps label token IDs to their index in the active set before
     computing cross-entropy.
3. Training proceeds normally. Checkpoints are saved with the **full** vocabulary weight
   matrix, so conversion to HuggingFace format is unaffected.

The input embedding layer is not modified — all input tokens are still processed.

## Loss Value Changes

```{warning}
Enabling `answer_only_vocab` changes the loss value. Cross-entropy is computed over
**N** classes (the answer vocab subset) instead of the full **V** classes, which removes
inactive tokens from the softmax denominator. Absolute loss numbers are **not comparable**
between a baseline run and a run with `answer_only_vocab = True`.

This is a fundamental trade-off: exact cross-entropy and reduced output-matmul compute
are mutually exclusive. Use this feature when training speed matters more than loss
comparability with a full-vocab baseline.
```

## Configuration

Set `answer_only_vocab = True` in `FinetuningDatasetConfig`:

```python
from megatron.bridge.training.config import (
    ConfigContainer,
    FinetuningDatasetConfig,
)

config = ConfigContainer(
    dataset=FinetuningDatasetConfig(
        train_data_path=[...],
        answer_only_loss=True,
        answer_only_vocab=True,              # enable vocab slicing
        answer_only_vocab_max_samples=10000, # optional: limit dataset scan
    ),
    # ... other config
)
```

`answer_only_vocab_max_samples` limits how many training samples are scanned to collect
the active vocabulary. `None` (the default) scans the full training dataset. For very large
datasets, setting a limit (e.g. `10000`) is usually sufficient to capture the complete answer
vocabulary while keeping startup time short.

## Parallelism Support

| Mode | Support |
|------|---------|
| Single GPU | Supported |
| DDP | Supported |
| Float16Module | Supported |
| PP > 1 | Supported — non-post-process stages are silently skipped; only the final stage (which owns the output layer) is patched. |
| TP > 1 | **Not supported.** With tensor parallelism, the output layer weight is a local shard of shape `[V//TP, H]`. Indexing it with global active IDs produces incorrect logits on ranks other than 0. When TP > 1 is detected, installation is skipped and a warning is logged; training proceeds with the full vocabulary. |
| FSDP | Not tested. |

## Measured Speedup

The speedup scales with the ratio `V / N`. Results on a 60M-parameter model with
`SEQ=16384`, packed sequences, single GPU:

| Vocabulary | Step time | TFLOP/s |
|-----------|-----------|---------|
| Full (64,260 tokens) | 5.41 s/step | 31.6 |
| Sliced (260 tokens) | 3.03 s/step | 55.8 |
| **Speedup** | **1.79×** | |

The speedup for larger models is similar because the output projection and loss scale
with `V`, which is model-independent.

## Low-Level API

The feature is wired automatically when `answer_only_vocab = True`. For custom forward
steps or non-standard training loops you can call the underlying functions directly:

```python
from megatron.bridge.training.vocab_slice import (
    collect_active_vocab_ids,
    create_vocab_sliced_forward_step,
)

# Collect unique answer token IDs from the training dataset
active_ids = collect_active_vocab_ids(train_ds, max_samples=10000)

# Wrap your forward step — vocab slice is installed lazily on first call
fwd_step = create_vocab_sliced_forward_step(active_ids, base_forward_step=my_fwd_step)

finetune(config=cfg, forward_step_func=fwd_step)
```

`create_vocab_sliced_forward_step` installs the slice **lazily** on the first call, so it
works correctly even when the model is reconstructed across in-process restart iterations.
