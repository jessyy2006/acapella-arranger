# Spec — Baseline Model

## Goal

Ship `src/models/baseline.py`: a pure-Transformer seq2seq model that consumes the same batches as `SATBHybrid` and produces the same shape of output, with **no LSTM anywhere**. This is the ablation comparison — "does the LSTM decoder actually help vs. a standard Transformer decoder?"

## Rubric justification

- **Baseline model for comparison** — 3 pts. Specifically requires that the baseline be trained and evaluated on the same data as the main model.
- **Distinct architecture for this modality** — this baseline itself doesn't qualify (the hybrid does), but your baseline is what makes the hybrid's "custom architecture combining paradigms" claim measurable.

## Interface contract

File: `src/models/baseline.py`

```python
from src.data.vocab import PAD, VOCAB_SIZE

class SATBBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        pad_idx: int = PAD,
        d_model: int = 256,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None: ...

    def encode(self, lead: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (memory, pad_mask). Same contract as SATBHybrid.encode."""

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Input:  the dict that src.data.loaders.collate_satb emits.
        Output: {"s": Tensor, "a": Tensor, "t": Tensor, "b": Tensor}
                where each value has shape (B, L_voice, vocab_size).
        """

    def num_parameters(self) -> int:
        """Total trainable parameters."""
```

Must be a drop-in replacement for `SATBHybrid` — training and evaluation code will type-hint `nn.Module` and expect this exact `forward` contract.

## Dependencies to import

- `src.data.vocab` — `PAD`, `VOCAB_SIZE`
- `torch.nn` — `TransformerEncoder`, `TransformerEncoderLayer`, `TransformerDecoder`, `TransformerDecoderLayer`, `Embedding`, `Linear`, `Dropout`
- Feel free to copy `SinusoidalPositionalEncoding` from `src/models/hybrid.py` (small class, fine to duplicate)

**Do not import** any custom code from `src/models/hybrid.py` — the baseline should stand on its own so ablations compare cleanly.

## Required tests

Add a new class `TestBaseline` to `tests/test_models.py` (don't create a new file — keeps coverage contiguous). Cover at minimum:

1. `test_returns_logits_for_all_four_voices` — same structure as `TestForwardShape` for the hybrid.
2. `test_logits_match_target_length_per_voice` — `(B, L_voice, VOCAB_SIZE)` for each voice.
3. `test_logits_are_finite` — no NaN/Inf on a random batch.
4. `test_padding_invariance` — extra PAD tokens at the tail of the lead don't change encoder outputs at valid positions. (Use the same pattern as the hybrid's test.)
5. `test_is_causal_on_decoder` — **this is the key one the hybrid test doesn't have, because the hybrid's unidirectional LSTM is causal by construction**. For the baseline, the decoder self-attention needs an explicit causal mask. Test: change a future token in a target voice and verify that logits at earlier positions are unchanged (bitwise).
6. `test_param_count_is_comparable_to_hybrid` — should be within ±50% of `SATBHybrid().num_parameters()` so the ablation isn't confounded by one model being dramatically larger.
7. `test_integration_with_real_collate` — feed a `collate_satb` batch from a real `SATBDataset` and assert output shape.

Mirror `tests/test_models.py` style — `pytest.fixture` for a tiny eval-mode model, helper `_make_synthetic_batch` (or import it from the existing file if useful).

## Design freedom

You decide:

1. **One decoder stack or four?** The hybrid has four separate decoder heads. The baseline can either:
   - Have four separate `TransformerDecoder` stacks (one per voice) — most directly comparable to the hybrid.
   - Have one shared `TransformerDecoder` stack with a small per-voice conditioning embedding prepended to the target — more parameter-efficient, a different ablation story.

   **Recommended default**: four separate stacks. Keeps the hybrid-vs-baseline comparison clean (only LSTM vs Transformer self-attn changes).

2. **Share lead embedding across voices or duplicate?** The hybrid shares. Baseline should too — keeps the lead-side comparison apples-to-apples.

3. **Tie input and output embeddings?** Optional. Mention your choice in the PR description.

4. **Target-side positional encoding**: sinusoidal is fine. If you want learned positional embeddings, explain why in the PR.

## Gotchas

- **Causal mask** is non-negotiable. `nn.Transformer.generate_square_subsequent_mask(L).to(device)` gives you the `(L, L)` upper-triangular `-inf` mask. Pass it as `tgt_mask=...` to your decoder layer's forward.
- **Two padding masks** flow through the decoder: `tgt_key_padding_mask` (target's own PAD positions) and `memory_key_padding_mask` (lead's PAD positions, from the encoder). Don't swap them.
- **`batch_first=True`** on every TransformerLayer. The hybrid uses this. Staying consistent keeps batch dims sane.
- **Embedding scaling by √d_model** — the hybrid does this. Skipping it in the baseline is a confound. Match the hybrid.
- **`norm_first=True` (pre-norm)** — hybrid uses this for training stability in deeper stacks. Use the same in the baseline. Ignore the benign `enable_nested_tensor` UserWarning PyTorch emits.
- **Padding index on `nn.Embedding`** — always pass `padding_idx=pad_idx`. Zeros out its gradient. Prevents the model learning on PAD tokens.

## Files to read first

Before writing any code:

1. **`src/models/hybrid.py`** — your contract reference. Same docstring format, same constructor shape, same forward signature.
2. **`tests/test_models.py`** — the test patterns you'll mirror. Pay attention to `_make_synthetic_batch` and the fixtures.
3. **`src/data/loaders.py`** — specifically `collate_satb` at the bottom. That's the dict your `forward` consumes.
4. **`src/data/vocab.py`** — PAD=0, VOCAB_SIZE=141. Short file; read all of it.

## Acceptance criteria

- [ ] All 7 required tests pass.
- [ ] Full `pytest` suite stays green.
- [ ] `num_parameters()` returns within ±50% of `SATBHybrid().num_parameters()` (~7.8M as of Day 3 Chunk 1).
- [ ] Module docstring explains: one-stack-vs-four choice, embedding tying choice, any other design decisions.
- [ ] PR description includes: parameter count, wall-clock forward-pass time on a `(4, 64)` batch (rough benchmark), and a one-paragraph "how this will compare to the hybrid" note.

## Out of scope

- **Training this model** — that's a training-runs task that happens after Jess ships the training loop. You're only building the architecture + tests right now.
- **Evaluating this model** — your `src/eval/*` spec covers that separately.
