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

An AI coding assistant (Claude Code) was used during implementation as a programming aid. Per course policy, a substantive account follows.

**Project team owned (not AI-generated):**
- Product requirements, scope, and system design (see `PRD.md`).
- Choice of pipeline decomposition (Demucs → CREPE → harmony model → export).
- Custom architecture decision (hybrid Transformer + LSTM vs alternatives).
- Two-stage training strategy (JSB Chorales pretrain → jaCappella fine-tune).
- Data augmentation strategy (×12 key transposition, sliding-window chunking).
- Voice-leading rule specification (vocal ranges, forbidden intervals).
- Evaluation metric definitions and their musical justification.
- Ablation design (which axes to vary and why).
- Interpretation of training curves, error cases, and ablation results.
- All claims in the README and walkthrough video.

**AI-assisted (reviewed, edited, and tested by the team before committing):**
- Repository scaffolding (directory layout, initial `requirements.txt`, doc skeletons).
- Implementation of utility functions once the team specified the contract (e.g., MIDI-to-tensor conversions, `yt-dlp` download wrappers, Streamlit layout boilerplate).
- Debugging assistance (tracing tensor shape mismatches, resolving dependency version conflicts).
- First drafts of documentation prose, subsequently reviewed and edited.

**Explicitly not AI-generated:**
- Evaluation numbers reported in the README.
- Conclusions drawn from error analysis.
- Claims about the model's strengths and failure modes in the walkthrough video.

The training loop, loss computation, custom model architecture, and evaluation pipeline were hand-reviewed line-by-line by the team.

## Academic References

- Rouard, S., Massa, F., & Défossez, A. (2023). *Hybrid Transformers for Music Source Separation.* ICASSP 2023.
- Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018). *CREPE: A Convolutional Representation for Pitch Estimation.* ICASSP 2018.
- Tamaru, T. et al. (2023). *jaCappella corpus: A Japanese a cappella vocal ensemble corpus.* ICASSP 2023.
- Hadjeres, G., Pachet, F., & Nielsen, F. (2017). *DeepBach: a Steerable Model for Bach Chorales Generation.* ICML 2017.
- Huang, C.-Z. A. et al. (2019). *Coconet: Counterpoint by Convolution.* ISMIR 2019.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
