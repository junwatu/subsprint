"""subsprint CLI: `subsprint transcribe` and `subsprint translate`."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="subsprint",
                                description="Offline subtitle generator for any video")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="transcribe video audio to .srt")
    t.add_argument("video", help="input video file")
    t.add_argument("-l", "--language", default=None,
                   help="audio language (e.g. en); omit for auto-detect")
    t.add_argument("-m", "--model", default="small",
                   help="whisper model: tiny/base/small/medium/large-v3-turbo (default: small)")
    t.add_argument("--backend", choices=["faster", "mlx"], default="faster",
                   help="faster-whisper (CPU) or mlx-whisper (Metal, M-series). default: faster")
    t.add_argument("--beam-size", type=int, default=1)
    t.add_argument("--vad", dest="vad",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="voice-activity filter (default: on)")
    t.add_argument("--device", default="cpu")
    t.add_argument("--compute-type", default="int8")
    t.add_argument("-o", "--output", default=None, help="output .srt path")
    t.add_argument("--output-dir", default=None)
    t.add_argument("--suffix", default=None, help="e.g. .en.auto (default: .<lang>.auto)")
    t.add_argument("--format", choices=["srt", "vtt"], default="srt")

    r = sub.add_parser("translate", help="translate an existing .srt, keep timings")
    r.add_argument("srt", help="input .srt file")
    r.add_argument("--from", dest="src", required=True, help="source lang code, e.g. it")
    r.add_argument("--to", dest="tgt", required=True, help="target lang code, e.g. en")
    r.add_argument("--model", default=None, help="HF model id (default: Helsinki-NLP/opus-mt-<src>-<tgt>)")
    r.add_argument("-o", "--output", default=None)
    r.add_argument("--batch-size", type=int, default=32)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "transcribe":
        from . import transcribe
        transcribe.run(args)
    elif args.cmd == "translate":
        from . import translate
        translate.run(args)


if __name__ == "__main__":
    main()
