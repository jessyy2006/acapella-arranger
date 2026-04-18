# Spec — Audio → Lead Tokens Pipeline

## Goal

Ship `src/pipeline/audio_to_midi.py`: given an audio file (MP3, WAV, M4A), return a token list in our vocabulary grammar representing the monophonic lead melody. This is the input side of `run_pipeline()` — everything downstream (the SATB model, voice-leading post-process, MIDI export) consumes your output directly.

## Rubric justification

Your pipeline is the audio half of the two-stage cross-modal system. Rubric items this closes (either standalone or by enabling other items):

- **Multi-stage ML pipeline** — 7 pts. Requires ≥2 distinct ML stages. Your stage: Demucs (vocal isolation) + torchcrepe (pitch estimation). Pairs with Jess's harmony model for the full pipeline.
- **Cross-modal generation** — 7 pts. Audio → symbolic tokens is the cross-modal bridge.
- **Audio preprocessing** — 7 pts. Spectrograms / pitch tracking / quantisation all count.
- **Explained architecture of pretrained model used** — 3 pts. You'll write a module docstring explaining HT-Demucs architecture; that's the rubric evidence.

## Interface contract

File: `src/pipeline/audio_to_midi.py`

```python
from pathlib import Path

def extract_lead_tokens(
    audio_path: str | Path,
    *,
    tempo_bpm: float | None = None,
    device: str | None = None,
) -> list[int]:
    """
    Load audio, isolate the lead vocal via HT-Demucs, pitch-track it with
    torchcrepe, quantise to the 16th-note grid, and encode to the vocab.

    Parameters
    ----------
    audio_path
        Path to an MP3 / WAV / M4A / FLAC file.
    tempo_bpm
        Target tempo for quantisation. If None, estimate with librosa.beat.beat_track.
    device
        "cuda", "mps", or "cpu". If None, auto-pick the best available.

    Returns
    -------
    A list of token ids conforming to src.data.vocab. Framed with SOS at
    the start, EOS at the end, BAR tokens at measure boundaries, REST
    tokens for silence.
    """
```

Also expose these smaller functions for testability (the big one above composes them):

```python
def isolate_vocals(audio_path: str | Path, device: str) -> tuple[np.ndarray, int]:
    """HT-Demucs. Returns (vocal_audio, sample_rate)."""

def pitch_track(
    audio: np.ndarray,
    sample_rate: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """torchcrepe. Returns (times_sec, pitch_hz, confidence) — all 1D, same length."""

def frames_to_part(
    times_sec: np.ndarray,
    pitch_hz: np.ndarray,
    confidence: np.ndarray,
    tempo_bpm: float,
    *,
    confidence_threshold: float = 0.5,
) -> "music21.stream.Part":
    """Collapse a frame-level pitch track into a quantised monophonic Part."""
```

The final `extract_lead_tokens` is `frames_to_part(*pitch_track(*isolate_vocals(...))) |> encode_part`.

## Dependencies to import

- **Existing in repo**: `src.data.tokenizer.encode_part`, `src.data.vocab` (constants, but you won't usually need them directly — encode_part uses them).
- **From `requirements.txt`**: `demucs`, `torchcrepe`, `librosa`, `soundfile`, `music21`, `numpy`, `torch`.
- **Don't re-implement**: silence detection — use `librosa.effects.split` with a reasonable `top_db`. Tempo estimation — use `librosa.beat.beat_track`. Pitch quantisation to MIDI — use `librosa.hz_to_midi` or equivalent.

## Required tests

New file `tests/test_audio_pipeline.py`. Minimum:

1. `test_synthesised_sine_round_trips` — generate a 2-second 440 Hz sine as a WAV (scipy.io.wavfile), feed through `extract_lead_tokens`, decode via `decode_part`, assert at least one note near MIDI 69 (A4). This is the "does it work at all" smoke test. Keep short — use small sample rate (e.g., 16 kHz) to stay fast.
2. `test_output_is_valid_vocab` — every returned token is in `[0, VOCAB_SIZE)`. Use the same sine-wave fixture.
3. `test_silence_produces_rest_tokens` — a silent input (array of zeros) should produce mostly `REST` tokens, not noise pitches.
4. `test_rejects_missing_file` — non-existent path raises `FileNotFoundError`.
5. `test_extracts_tempo_when_none_given` — pass `tempo_bpm=None`, assert the function ran and produced *some* output (don't assert a specific BPM — librosa's estimate on synthetic data is unreliable).

Gate the Demucs-heavy tests behind a `pytest.mark.slow` marker if the full download takes too long in CI — but smoke tests must run in the default `pytest` invocation. Use a cached Demucs model (`demucs.pretrained.get_model`) so the weights download only happens once.

## Design freedom

1. **Which Demucs model?** `htdemucs` (default) or `htdemucs_ft`. `htdemucs` is 80 MB; `htdemucs_ft` is ~330 MB but slightly better on vocals. Recommend `htdemucs` unless you observe bad vocal isolation in manual tests.
2. **torchcrepe model size** — `tiny` (~5 MB, fastest), `small`, `medium`, `full` (~90 MB, most accurate). Recommend `full` for quality, `tiny` for CI.
3. **Pitch smoothing** — raw frame-level torchcrepe output is jittery. Options:
   - Median filter with a 5-frame window on the MIDI-quantised pitch track.
   - Viterbi decoding over pitch confidences (torchcrepe has this built-in).
   - Just take the mode over each 16th-note bucket.

   Recommend median filter first, Viterbi if output is still jittery.
4. **Silence threshold on confidence** — torchcrepe returns a confidence per frame. Below some threshold (suggest 0.5), emit REST. Tune against the synthesised sine test.
5. **Tempo handling** — quantising to the 16th grid requires tempo. If the user's song is 120 BPM, one 16th-note is 125 ms. Ballpark: for an unknown audio, estimate tempo → round to nearest integer → quantise. This is lossy; that's fine.

## Gotchas

- **Demucs weight download is ~80–330 MB the first time.** Subsequent calls use the cache. Document this in the module docstring so users aren't surprised.
- **torchcrepe expects 16 kHz mono audio.** HT-Demucs outputs 44.1 kHz stereo. You need to resample + downmix before pitch tracking: `librosa.resample(librosa.to_mono(vocal), orig_sr=44100, target_sr=16000)`.
- **GPU memory**: on a T4 (16 GB) Demucs can handle ~2 min of audio. For longer inputs, chunk and stitch. Leave a TODO if chunking isn't implemented in v1.
- **Pitch at attack/decay**: torchcrepe confidence is low at note onsets and releases. Don't just trust mean pitch over a window — weight by confidence or drop low-confidence frames.
- **Octave errors**: torchcrepe occasionally reports pitches one octave off. Median filter helps. Worth flagging as a known limitation in your module docstring.
- **Drums leaking into the "vocals" stem**: HT-Demucs isn't perfect; snare hits can leak. If you see spurious pitches near the drum transient, consider a minimum-duration filter (drop "notes" shorter than ~1 16th at target tempo).
- **Don't assume the input audio is music.** A YouTube video might include talking intros. Your pipeline should degrade gracefully on non-pitched segments (→ RESTs).

## Files to read first

1. **`src/data/tokenizer.py`** — specifically `encode_part` and `decode_part`. You're producing input to `encode_part` (a monophonic `music21.stream.Part`), and your tests will use `decode_part` to round-trip.
2. **`src/data/vocab.py`** — token layout, duration buckets. Your quantisation grid is 16ths (`DUR_BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16)`).
3. **`notebooks/02_tokenizer_sanity.ipynb`** — end-to-end example of building a `stream.Part`, setting `partName`, and calling `encode_part`. Good template.
4. **`requirements.txt`** — exact pinned versions of demucs, torchcrepe, librosa.
5. **External**: [torchcrepe README](https://github.com/maxrmorrison/torchcrepe) (short), [HT-Demucs paper](https://arxiv.org/abs/2211.08553) (skim for architecture paragraph — rubric wants you to explain it in prose).

## Acceptance criteria

- [ ] All 5 required tests pass.
- [ ] Full `pytest` suite stays green.
- [ ] Given a real pop song MP3 (~30s clip), `extract_lead_tokens` returns tokens that `decode_part` renders to a MIDI file that *audibly resembles* the original lead melody when opened in GarageBand. Jess will verify this on her Mac.
- [ ] Module docstring includes a 1-paragraph explanation of HT-Demucs architecture (sourced for rubric evidence).
- [ ] Any known limitations (octave errors, drum leakage, etc.) are listed in the module docstring with the same "Known limitations" heading `src/data/tokenizer.py` uses.

## Out of scope

- **The end-to-end `run_pipeline(audio) -> editable MIDI` function** — Jess owns that. You own only the audio → token-list half.
- **Any voice-leading work** — that's Jess's `src/postprocess/voice_leading.py`.
- **Any Streamlit integration** — Jess owns the UI layer.
- **Training your own pitch model** — we use torchcrepe frozen. No training.
