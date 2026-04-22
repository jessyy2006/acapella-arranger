"""Aca-Adapt Streamlit web app.

Single-page interface: upload an audio file, run the end-to-end
``run_pipeline`` arranger, download the resulting SATB MIDI.

Designed to be deployed (Streamlit Community Cloud or equivalent)
as well as run locally via ``streamlit run src/app/main.py``. The
pipeline is CPU/MPS-bound and takes 1–3 minutes per clip on a
laptop; progress is surfaced via an ``st.status`` block so the user
doesn't see a dead UI during generation.

Heavy steps:

* Demucs vocal separation (~20-30 s for a 60 s clip on CPU)
* CREPE pitch tracking (~20 s on CPU, <5 s on MPS/CUDA)
* Autoregressive SATB decode per section (~5-15 s total)

The checkpoint path is read from the ``ACA_ADAPT_CHECKPOINT`` env
variable when set, otherwise falls back to
``checkpoints/phase_b/phase_b_final.pt`` relative to the project
root. This keeps the default setup zero-config for the dev laptop
while letting deployment override it.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "phase_b" / "phase_b_final.pt"
_CHECKPOINT = Path(os.environ.get("ACA_ADAPT_CHECKPOINT", _DEFAULT_CHECKPOINT))

_ACCEPTED_AUDIO = ["mp3", "wav", "m4a", "mp4", "flac", "ogg"]

logger = logging.getLogger(__name__)


def _inject_css() -> None:
    """Custom CSS for a cleaner, calmer layout than Streamlit defaults.

    Narrows the content column, tightens the heading hierarchy, adds
    generous whitespace around cards, and softens the default button
    treatment so the single primary CTA is unambiguous.
    """
    st.markdown(
        """
        <style>
          /* Constrain content width — Streamlit's wide mode is too wide
             for a single-column upload flow. */
          .block-container {
              max-width: 780px;
              padding-top: 3.5rem;
              padding-bottom: 4rem;
          }

          /* Tighter, more confident headings. */
          h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
          h2 { font-weight: 600 !important; letter-spacing: -0.01em; }

          /* Hero subtitle: muted, one size down. */
          .aca-subtitle {
              color: #64748B;
              font-size: 1.0625rem;
              line-height: 1.55;
              margin: -0.5rem 0 2rem 0;
          }

          /* Card treatment for the upload + results panels. */
          .aca-card {
              background: #FFFFFF;
              border: 1px solid #E2E8F0;
              border-radius: 16px;
              padding: 1.5rem 1.75rem;
              margin-bottom: 1.25rem;
              box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
          }

          /* Stat row — small labels above larger values. */
          .aca-stat-label {
              color: #64748B;
              font-size: 0.8125rem;
              text-transform: uppercase;
              letter-spacing: 0.04em;
              margin-bottom: 0.1rem;
          }
          .aca-stat-value {
              color: #0F172A;
              font-size: 1.25rem;
              font-weight: 600;
          }

          /* Primary button — fill the width of its column. */
          div.stButton > button[kind="primary"] {
              width: 100%;
              border-radius: 10px;
              font-weight: 600;
              padding: 0.6rem 1.2rem;
          }

          /* File uploader — subtler border, rounded. */
          section[data-testid="stFileUploader"] > div {
              border-radius: 12px;
              border: 1.5px dashed #CBD5E1;
              background: #FFFFFF;
          }

          /* Hide the default Streamlit footer + menu for a cleaner look. */
          footer { visibility: hidden; }
          #MainMenu { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=8).hexdigest()


@st.cache_data(show_spinner=False, max_entries=4)
def _probe_audio(audio_bytes: bytes, suffix: str) -> tuple[float, int]:
    """Return ``(duration_sec, sample_rate)`` for the uploaded audio.

    Cached by ``audio_bytes`` so repeat toggles of the advanced-options
    panel don't re-decode the file. ``suffix`` is part of the cache key
    so two clips that happen to share bytes-length stay distinct.
    """
    import librosa

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        return float(len(y) / sr if sr > 0 else 0.0), int(sr)
    finally:
        os.unlink(tmp_path)


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _run_arrangement(
    audio_bytes: bytes,
    original_name: str,
    voice_leading: bool,
    tempo_override: float | None,
) -> tuple[bytes, dict[str, float | int | str]]:
    """Execute ``run_pipeline`` on ``audio_bytes`` and return
    ``(midi_bytes, info)`` where ``info`` holds values shown on the
    results card (detected tempo, target sixteenths, elapsed time).

    Streamlit's ``st.status`` is used for progress messaging because
    the pipeline doesn't natively expose intermediate events —
    wrapping each major call in its own status line gives the user a
    sense of forward motion rather than a spinner stalled on the
    Demucs download.
    """
    # Imported lazily so the app boots quickly when nothing's uploaded.
    from src.pipeline.run_pipeline import run_pipeline

    if not _CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {_CHECKPOINT}. "
            "Set the ACA_ADAPT_CHECKPOINT environment variable or place the "
            "phase_b_final.pt file at the default location."
        )

    suffix = Path(original_name).suffix or ".mp3"
    t0 = time.monotonic()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as in_tmp:
        in_tmp.write(audio_bytes)
        in_path = Path(in_tmp.name)
    out_path = Path(tempfile.mkdtemp()) / f"{Path(original_name).stem}_arrangement.mid"

    try:
        run_pipeline(
            audio_path=in_path,
            model_checkpoint=_CHECKPOINT,
            out_path=out_path,
            tempo_bpm=tempo_override,
            voice_leading=voice_leading,
        )
        midi_bytes = out_path.read_bytes()
    finally:
        # Cleanup: input was a temp; output MIDI bytes are already in
        # memory so we can delete the file too.
        try:
            in_path.unlink()
        except FileNotFoundError:
            pass
        try:
            out_path.unlink()
            out_path.parent.rmdir()
        except FileNotFoundError:
            pass

    elapsed = time.monotonic() - t0
    return midi_bytes, {"elapsed_sec": elapsed}


def _render_hero() -> None:
    st.markdown(
        """
        <h1 style="margin-bottom: 0.25rem;">Aca-Adapt</h1>
        <p class="aca-subtitle">
          Turn any song into a four-part a cappella arrangement.
          Upload audio, get back a MIDI score — Soprano, Alto, Tenor, Bass —
          ready to open in MuseScore or any DAW.
        </p>
        """,
        unsafe_allow_html=True,
    )


def _render_file_info(duration_sec: float, size_bytes: int, filename: str) -> None:
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            f'<div class="aca-stat-label">File</div>'
            f'<div class="aca-stat-value" style="font-size:1rem;">{filename}</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="aca-stat-label">Duration</div>'
            f'<div class="aca-stat-value">{_format_duration(duration_sec)}</div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        mb = size_bytes / (1024 * 1024)
        st.markdown(
            f'<div class="aca-stat-label">Size</div>'
            f'<div class="aca-stat-value">{mb:.1f} MB</div>',
            unsafe_allow_html=True,
        )


def _render_results(
    midi_bytes: bytes,
    output_filename: str,
    elapsed_sec: float,
    input_duration_sec: float,
) -> None:
    st.markdown("### Your arrangement is ready")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f'<div class="aca-stat-label">Input length</div>'
            f'<div class="aca-stat-value">{_format_duration(input_duration_sec)}</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="aca-stat-label">Generated in</div>'
            f'<div class="aca-stat-value">{elapsed_sec:.1f} s</div>',
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.download_button(
        label="Download MIDI",
        data=midi_bytes,
        file_name=output_filename,
        mime="audio/midi",
        type="primary",
        use_container_width=True,
    )
    st.caption(
        "The output is a five-track MIDI: Lead (extracted from your upload) + "
        "Soprano, Alto, Tenor, Bass. Every track uses General MIDI program 0 "
        "(Acoustic Grand Piano) so it opens cleanly in any DAW."
    )


def main() -> None:
    st.set_page_config(
        page_title="Aca-Adapt",
        page_icon="🎼",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _inject_css()
    _render_hero()

    # ---- Upload card -------------------------------------------------
    uploaded = st.file_uploader(
        "Upload an audio file",
        type=_ACCEPTED_AUDIO,
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.caption(
            "Supported formats: MP3, WAV, M4A, MP4, FLAC, OGG. "
            "Up to 200 MB, ideally 30 s – 3 min."
        )
        return

    audio_bytes = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower() or ".mp3"

    try:
        duration_sec, _sample_rate = _probe_audio(audio_bytes, suffix=suffix)
    except Exception as exc:
        st.error(f"Couldn't read that audio file: {exc}")
        return

    if duration_sec < 5.0:
        st.warning(
            "That file is under 5 seconds — the pipeline needs a longer clip "
            "to track pitch and detect sections reliably."
        )
        return

    _render_file_info(duration_sec, len(audio_bytes), uploaded.name)

    # Audio preview directly under the info card — gives the user a
    # quick sanity check that the upload decoded as they expected.
    # Let Streamlit auto-detect the format from the bytes; passing a
    # bad MIME hint (e.g. ``audio/mp4`` for a multi-codec container)
    # can confuse the browser's playback element.
    st.audio(audio_bytes)

    # ---- Advanced options -------------------------------------------
    with st.expander("Advanced options", expanded=False):
        tempo_override = st.number_input(
            "Tempo override (BPM)",
            min_value=0.0,
            max_value=300.0,
            value=0.0,
            step=0.5,
            help=(
                "Leave at 0 to auto-detect tempo from the audio. "
                "Set a specific BPM if you know the tempo and auto-detection "
                "is landing on a wrong multiple (common on syncopated pop)."
            ),
        )
        voice_leading = st.checkbox(
            "Voice-leading post-process",
            value=True,
            help=(
                "Range-clamps each voice to its human singing range, "
                "coalesces same-pitch neighbours, and aligns event ends to "
                "a shared 8th-note grid so collective pauses line up. "
                "Recommended on."
            ),
        )

    # ---- Generate ----------------------------------------------------
    if not _CHECKPOINT.is_file():
        st.error(
            "Model checkpoint is missing at the expected location: "
            f"`{_CHECKPOINT}`. Set `ACA_ADAPT_CHECKPOINT` to point at "
            "`phase_b_final.pt` and rerun."
        )
        return

    go = st.button("Generate arrangement", type="primary", use_container_width=True)
    if not go:
        return

    # Memoise the generated MIDI by (file hash, options) so re-clicking
    # Generate on the same inputs returns immediately. Streamlit's
    # session_state is the right lifetime here — cache_data would
    # persist across users in a multi-user deployment.
    cache_key = (
        _hash_bytes(audio_bytes),
        tempo_override,
        voice_leading,
    )
    cached = st.session_state.get("_result", {})
    if cached.get("key") != cache_key:
        with st.status(
            "Generating arrangement… (typically 1–3 minutes on CPU)",
            expanded=True,
        ) as status:
            # run_pipeline is opaque — these lines document the stages
            # the user is waiting on rather than tracking live progress.
            status.write("• Isolating vocals with HT-Demucs")
            status.write("• Tracking pitch with CREPE")
            status.write("• Generating SATB with the trained hybrid model")
            status.write("• Voice-leading + fit to input duration")
            try:
                midi_bytes, info = _run_arrangement(
                    audio_bytes,
                    original_name=uploaded.name,
                    voice_leading=voice_leading,
                    tempo_override=tempo_override if tempo_override > 0 else None,
                )
            except FileNotFoundError as exc:
                status.update(label="Checkpoint missing", state="error")
                st.error(str(exc))
                return
            except RuntimeError as exc:
                status.update(label="Pipeline error", state="error")
                st.error(f"Pipeline failed: {exc}")
                return
            except Exception as exc:  # pragma: no cover — belt-and-braces
                logger.exception("pipeline failed")
                status.update(label="Unexpected error", state="error")
                st.error(f"Unexpected error: {exc}")
                return
            status.update(label="Done", state="complete")
        st.session_state["_result"] = {
            "key": cache_key,
            "midi_bytes": midi_bytes,
            "elapsed_sec": info["elapsed_sec"],
            "input_duration_sec": duration_sec,
            "filename": f"{Path(uploaded.name).stem}_arrangement.mid",
        }

    result = st.session_state["_result"]
    _render_results(
        midi_bytes=result["midi_bytes"],
        output_filename=result["filename"],
        elapsed_sec=result["elapsed_sec"],
        input_duration_sec=result["input_duration_sec"],
    )


if __name__ == "__main__":
    main()
