"""Transcribe any video to SRT using faster-whisper or mlx-whisper (offline)."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pysubs2


def check_ffmpeg() -> str:
    import shutil
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not found. Install with: brew install ffmpeg")
    return exe


def extract_audio(video: Path, wav: Path) -> None:
    cmd = [
        check_ffmpeg(), "-y", "-v", "error",
        "-i", str(video),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(wav),
    ]
    subprocess.run(cmd, check=True)


def default_out(video: Path, lang: str | None, suffix: str | None, out_dir: Path | None) -> Path:
    name = video.stem + (suffix or (f".{lang}.auto" if lang else ".auto"))
    return (out_dir or video.parent) / (name + ".srt")


def transcribe_faster(wav: Path, *, model: str, language: str | None,
                      beam_size: int, vad: bool, device: str,
                      compute_type: str):
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device=device, compute_type=compute_type)
    vad_params = dict(min_silence_duration_ms=500) if vad else None
    segments, info = m.transcribe(
        str(wav), language=language, beam_size=beam_size,
        vad_filter=vad, vad_parameters=vad_params,
    )
    return segments, getattr(info, "language", language), getattr(info, "language_probability", None)


def transcribe_mlx(wav: Path, *, model: str, language: str | None):
    import mlx_whisper
    repo = f"mlx-community/whisper-{model}-mlx"
    kwargs: dict = {"verbose": False}
    if language:
        kwargs["language"] = language
    res = mlx_whisper.transcribe(str(wav), path_or_hf_repo=repo, **kwargs)
    for s in res.get("segments", []):
        yield type("S", (), {"start": s["start"], "end": s["end"], "text": s["text"]})


def run(args) -> Path:
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    out = Path(args.output) if args.output else default_out(
        video, args.language, args.suffix,
        Path(args.output_dir) if args.output_dir else None,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = (args.format or "srt").lower()

    with tempfile.TemporaryDirectory(prefix="subgen-") as td:
        wav = Path(td) / "audio16k.wav"
        print(f"[subgen] extracting audio: {video.name} -> 16kHz mono", flush=True)
        extract_audio(video, wav)
        print(f"[subgen] transcribing with {args.backend}/{args.model} "
              f"(lang={args.language or 'auto'}, beam={args.beam_size}, vad={args.vad}) ...", flush=True)
        if args.backend == "mlx":
            segments = transcribe_mlx(wav, model=args.model, language=args.language)
            subs = pysubs2.SSAFile()
            for s in segments:
                subs.append(pysubs2.SSAEvent(start=int(s.start * 1000),
                                             end=int(s.end * 1000),
                                             text=(s.text or "").strip()))
        else:
            segments, det_lang, det_prob = transcribe_faster(
                wav, model=args.model, language=args.language,
                beam_size=args.beam_size, vad=args.vad,
                device=args.device, compute_type=args.compute_type,
            )
            print(f"[subgen] detected language: {det_lang} ({det_prob})", flush=True)
            subs = pysubs2.SSAFile()
            for s in segments:
                subs.append(pysubs2.SSAEvent(start=int(s.start * 1000),
                                             end=int(s.end * 1000),
                                             text=(s.text or "").strip()))
        print(f"[subgen] segments: {len(subs)}", flush=True)
        if fmt == "vtt":
            out = out.with_suffix(".vtt")
        subs.save(str(out))
    print(f"[subgen] saved: {out}", flush=True)
    return out
