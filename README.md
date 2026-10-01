# subsprint — offline subtitle generator for any video

CLI that creates `.srt` subtitles from a video file, fully offline after the
first model download. Tuned for Apple Silicon (M2 Pro tested).

## Install

```bash
cd ~/Developers/projects/subsprint
pip install -e .
# optional, faster on M-series via Metal:
pip install -e ".[mlx]"
# optional, for `translate` command:
pip install -e ".[translate]"
```

Requires system `ffmpeg` (`brew install ffmpeg`).

## Use

```bash
# transcribe English audio to English subs
# (pass -l en for English audio, omit -l for auto-detect)
subsprint transcribe movie.mp4 -l en

# higher quality, still <10 min with Metal backend:
subsprint transcribe movie.mp4 -l en --backend mlx --model small

# best quality offline (slower on CPU, faster on Metal):
subsprint transcribe movie.mp4 -l en --model large-v3-turbo --beam-size 1

# custom output dir / suffix / format
subsprint transcribe movie.mp4 -l en -o ./subs --suffix .en.auto --format srt

# translate an existing .srt (e.g. Italian -> English), keeps timings
subsprint translate subs.srt --from it --to en -o subs.eng.srt
```

Outputs by default sit next to the video:
`<video-stem>.<lang>.auto.srt` (e.g. `movie.en.auto.srt`).

## Example timings (English subs, measured on M2 Pro 10-core / 16 GB)

Fast defaults: model `small`, `beam_size=1`, VAD on, `faster-whisper` CPU backend.

| Video length | Audio language | Model / backend | Subs generation time |
| --- | --- | --- | --- |
| 1 min | English | `small` / faster-whisper (CPU) | ~5 s transcribe (+ one-time model load) |
| 10 min | English | `small` / faster-whisper (CPU) | ~50 s (scales linearly) |
| 98 min | English | `small` / faster-whisper (CPU) | ~5 min transcribe + ~20 s audio extract, 1,474 segments |

Notes:

- First run downloads the model once (`small` ≈ 500 MB); afterwards 100 % offline.
- Model load is extra on first run only (~45 s with download, ~1 s cached).
- `tiny` is faster but noticeably less accurate; `medium` / `large-v3-turbo`
  are more accurate and slower (turbo is much faster on the Metal `mlx` backend).
- `mlx-whisper` (`--backend mlx`) uses the Metal GPU: ~3 s inference per
  60 s of audio with `small` after the one-time model download.
- `translate` uses `Helsinki-NLP/opus-mt-<src>-<tgt>`; first run downloads it too.
