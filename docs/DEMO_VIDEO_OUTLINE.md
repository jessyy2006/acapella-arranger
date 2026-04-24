# Demo Video Outline

Scratchpad of ideas for the two required videos. Fill in when we sit down to record.

## Production hardening demo (rubric #83)

Idea: record a split-screen of the deployed Space UI + HF's Container log tab. Narrate in real time.

Script beats:
1. Upload a short clip (30–60 s) — the container log prints `event=upload session=... filename=... size_mb=... uploads_total=1`. Point to the structured `key=value` format on screen.
2. Click Generate. The log prints `event=generate_start session=... duration_sec=... voice_leading=True`. Narrate: "same session id as the upload — that's how a grader can correlate a visitor's events without a metrics backend."
3. While the pipeline runs, click Generate three more times. The fourth click is rejected with `event=rate_limited session=... wait_sec=...`. Narrate: "sliding-window rate limit, configurable via env var, prevents a single visitor from saturating the free CPU tier."
4. When the first generation finishes, the log prints `event=generate_ok elapsed_sec=... gens_total=1 fails_total=0`. Narrate: "rolling counters, no external dashboard required."
5. (Optional, if you want to show failure handling) Upload a broken file — the log prints `event=probe_error error=CalledProcessError` and the UI shows a matching clean error message. Narrate: "classified error paths — each except branch logs a distinct reason and surfaces a matching UI message."

One minute of video covers all five pillars of #83: caching (session_state cache-hit returns instantly on repeat click), rate limiting (live rejection), structured logging (grep-able lines), monitoring (counters), and classified error handling.

## Everything else

_TBD — add intro, non-technical narrative beats, pipeline architecture visualization, and closing before recording._
