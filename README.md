# subgen — offline subtitle generator for any video

CLI that creates `.srt` subtitles from a video file, fully offline after the
first model download. Tuned for Apple Silicon (M2 Pro tested).

## Install

```bash
cd ~/Developers/projects/subgen
pip install -e .
# optional, faster on M-series via Metal:
pip install -e ".[mlx]"
# optional, for `translate` command:
pip install -e ".[translate]"
```

Requires system `ffmpeg` (`brew install ffmpeg`).

## Use

```bash
# transcribe (auto language -> English subs by default is NOT assumed;
# pass -l en for English audio, omit for auto-detect)
subgen transcribe movie.mp4 -l en

# fast defaults: model=small, beam=1, VAD on (~8 min for a 98 min film on M2 Pro)
# higher quality, still <10 min with Metal backend:
subgen transcribe movie.mp4 -l en --backend mlx --model small

# best quality offline (slower, ~40-60 min on CPU, faster on Metal):
subgen transcribe movie.mp4 -l en --model large-v3-turbo --beam-size 1

# custom output dir / suffix / format
subgen transcribe movie.mp4 -l en -o ./subs --suffix .en.auto --format srt

# translate an existing .srt (e.g. Italian -> English), keeps timings
subgen translate Subs/SDH.ita.HI.srt --from it --to en -o Subs/SDH.eng.HI.srt
```

Outputs by default sit next to the video:
`<video-stem>.<lang>.auto.srt` (e.g. `Runner...en.auto.srt`).

## Notes

- First run downloads the model (~500MB for `small`), afterwards 100% offline.
- `small + beam 1 + VAD` ≈ 13x realtime on M2 Pro CPU (measured 4.6s per 60s audio).
- `translate` uses `Helsinki-NLP/opus-mt-<src>-<tgt>`; first run downloads it too.
