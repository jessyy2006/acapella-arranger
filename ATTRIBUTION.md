# Attribution

All third-party assets, datasets, and tools used in Aca-Adapt are credited below.

## Pre-trained Models

| Model | Authors / Source | Role in Aca-Adapt | License |
|---|---|---|---|
| **HT-Demucs** | Meta AI — Rouard et al., 2022 — https://github.com/facebookresearch/demucs | Vocal stem isolation (frozen, inference only) | MIT |
| **CREPE** | Kim, Salamon, Li, Bello (NYU), 2018 — https://github.com/marl/crepe | Monophonic pitch estimation (frozen, inference only) | MIT |

Both models are used as frozen, pretrained feature extractors via their official PyPI packages. No weights were modified.

## Datasets

| Dataset | Source | Role | Citation |
|---|---|---|---|
| **jaCappella corpus** | HuggingFace | Phase B fine-tuning (Japanese a cappella SATB) | Tamaru et al., 2023 (ICASSP) |
| **JSB Chorales** | `music21.corpus.chorales.Iterator()` | Phase A pretraining (Bach SATB chorales) | Public domain |
| **YouTube audio** (via `yt-dlp`) | User-supplied URLs at inference time | Real-world testing only — not used in training | Respective copyright holders |

## Open-Source Libraries

| Library | Role |
|---|---|
| [PyTorch](https://pytorch.org/) | Custom harmony model + training loop |
| [music21](https://web.mit.edu/music21/) | MIDI parsing, voice-leading validation, MusicXML export, JSB corpus access |
| [librosa](https://librosa.org/) | Audio preprocessing (spectrograms, resampling) |
| [Streamlit](https://streamlit.io/) | Web application framework |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Inference-time audio download |
| [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) | Evaluation visualization |

Full dependency list: `requirements.txt`.

## AI Development Tools

An AI coding assistant (Claude Code) was used during implementation as a programming aid.

**Project team owned (not AI-generated):**
- Product requirements, scope, and system design (see `docs/PRD.md`).
- Choice of pipeline decomposition (Demucs → CREPE → harmony model → export).
- Custom architecture decision (hybrid Transformer + LSTM vs alternatives).
- Two-stage training strategy (JSB Chorales pretrain → jaCappella fine-tune).
- Data augmentation strategy (×12 key transposition, sliding-window chunking).
- Voice-leading rule specification (vocal ranges, forbidden intervals).
- Evaluation metric definitions and their musical justification.
- Ablation design (which axes to vary and why).
- Interpretation of training curves, error cases, and ablation results.
- All claims in the README and walkthrough video.

**AI-assisted (built in collaboration with the team before committing):**
- Partial implementation of utility functions once the team specified the contract (e.g., MIDI-to-tensor conversions, `yt-dlp` download wrappers, Streamlit layout boilerplate).
- Debugging assistance (tracing tensor shape mismatches, resolving dependency version conflicts).

**Explicitly not AI-generated:**
- Evaluation numbers reported in the README.
- Conclusions drawn from error analysis.
- Claims about the model's strengths and failure modes in the walkthrough video.

The training loop, loss computation, custom model architecture, and evaluation pipeline were hand-reviewed line-by-line by the team.

### Implementation Log

Per-file record of Claude Code's contribution to each committed artifact, kept current as the project progresses. **Designed** = the team specified the contract, interfaces, and approach; Claude filled in implementation details against that spec. **Drafted** = Claude produced an initial version that the team reviewed, edited, and tested before commit. All files were hand-reviewed before each commit.

#### Day 1 (2026-04-18) — repo scaffold + data exploration

| File | Team ownership | Claude Code role |
|---|---|---|
| `docs/PRD.md` | product scope, pipeline architecture, 15-item rubric plan | section formatting, cross-reference checks |
| `scripts/download_data.py` | HuggingFace snapshot strategy, gated-dataset handling flow | drafted implementation |
| `notebooks/01_data_exploration.ipynb` | six analysis questions, filter criteria, clean-SATB definition | drafted cells; wrote `parse_lenient` and `is_clean_satb` helpers |
| `SETUP.md`, `README.md`, `PARTNER_BRIEF.md` | structure, prerequisites list, partner onboarding steps | prose drafting |
| `requirements.txt`, `.gitignore` | pin policy, ignore policy | drafted |

#### Day 2 (2026-04-18) — tokenizer + data pipeline

| File | Team ownership | Claude Code role |
|---|---|---|
| `src/data/load.py` | designed (promote notebook helpers into a module) | drafted |
| `src/data/vocab.py` | designed token layout (141-id space: 5 specials + 128 MIDI pitches + 8 duration buckets on the 16th-note grid) | implementation |
| `src/data/tokenizer.py` | designed (interleaved `PITCH/REST` + `DUR` pair grammar, BAR at measure boundaries, defensive chord-collapse fallback) | `encode_part` / `decode_part` implementation |
| `src/data/augmentation.py` | designed (×12 transposition range, 8-bar / 4-bar sliding windows, None-signals-skip contract for out-of-range shifts) | implementation |
| `src/data/dataset.py` | designed (per-source voice routing, pre-materialised augmentation, min-window alignment to prevent voice mis-pairing) | implementation |
| `src/data/loaders.py` | designed (70/15/15 by-song split to prevent leakage, train-only augmentation, per-voice padding collate with length tensors, `load_dataset` wrapper for PyTorch's `weights_only=True` default) | implementation |
| `scripts/prepare_data.py` | designed (argparse surface, idempotency contract, canonical-variant dedup for jaCappella's three-lyric-variant trap) | implementation |
| `tests/test_tokenizer.py`, `tests/test_augmentation.py`, `tests/test_dataset.py` (92 tests) | acceptance criteria, edge cases to cover, round-trip invariants | test code |
| `pytest.ini` | test discovery config | drafted |

Notable user-directed course corrections on Day 2:
- Catching that the jaCappella dataset ships each song as three musically identical MusicXML variants (base / romaji / SVS) that the first draft of `prepare_data.py` loaded indiscriminately — left unfixed, this would have triple-counted every song and leaked duplicates across the train/val/test split.
- Requiring that the lossy duration quantisation (tuplets and dotted-16ths snap to the nearest 16th-grid bucket) be flagged in both the `tokenizer.py` module docstring and the persistent scoring plan, so it surfaces in the final README's evaluation section rather than being silently absorbed.
- Catching the PyTorch ≥2.6 `weights_only=True` default in code review, before it blocked the Day 3 training-data load.

## Academic References

- Rouard, S., Massa, F., & Défossez, A. (2023). *Hybrid Transformers for Music Source Separation.* ICASSP 2023.
- Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018). *CREPE: A Convolutional Representation for Pitch Estimation.* ICASSP 2018.
- Tamaru, T. et al. (2023). *jaCappella corpus: A Japanese a cappella vocal ensemble corpus.* ICASSP 2023.
- Hadjeres, G., Pachet, F., & Nielsen, F. (2017). *DeepBach: a Steerable Model for Bach Chorales Generation.* ICML 2017.
- Huang, C.-Z. A. et al. (2019). *Coconet: Counterpoint by Convolution.* ISMIR 2019.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
