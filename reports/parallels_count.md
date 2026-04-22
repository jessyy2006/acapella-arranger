## Parallel-motion violation counts per generated sample

Generated 15 samples per (checkpoint, test split) with pitch-T=0.5, dur-T=1.1, top-k=10. For each generated SATB arrangement, we count how many time-steps the voice-leading detector flags as parallel 5ths or parallel 8ves between adjacent voices. Lower is better.

| run | mean | median | max | total |
|---|---:|---:|---:|---:|
| `baseline_phase_a_jsb` | 1.33 | 1.0 | 4 | 20 |
| `phase_a_jacappella` | 0.47 | 0.0 | 2 | 7 |
| `phase_a_jsb` | 0.40 | 0.0 | 2 | 6 |
| `phase_b_jacappella` | 0.07 | 0.0 | 1 | 1 |
| `phase_b_jsb` | 0.20 | 0.0 | 1 | 3 |

### Interpretation

A violation here is textbook-forbidden voice motion (two adjacent voices moving in the same direction while holding the same 5th or 8ve interval). The model was not explicitly trained to avoid these, so the count is a direct measure of how well the model internalised the constraint from the training data. The detector is implemented by `src.postprocess.voice_leading.detect_parallel_motion` and defined formally in that module's docstring.
