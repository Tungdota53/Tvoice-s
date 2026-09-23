"""Utility tool to register or verify local RVC ONNX model checkpoints."""

import argparse
import json
import shutil
import sys
from pathlib import Path

VOICES_DIR = Path(__file__).resolve().parent / "voices"


def register_voice(
    voice_id: str,
    name: str,
    gender: str,
    model_path: Path,
    hubert_path: Path,
    rmvpe_path: Path,
    pitch_shift: float = 0.0,
    desc: str = "Imported AI Voice",
):
    target_dir = VOICES_DIR / voice_id
    target_dir.mkdir(parents=True, exist_ok=True)

    for src, fname in [(model_path, "model.onnx"), (hubert_path, "hubert.onnx"), (rmvpe_path, "rmvpe.onnx")]:
        if not src.exists():
            print(f"[ERROR] Source file does not exist: {src}")
            sys.exit(1)
        dst = target_dir / fname
        print(f"[COPY] {src} -> {dst}")
        shutil.copy2(src, dst)

    cfg = {
        "name": name,
        "description": desc,
        "gender": gender,
        "category": "ai_voice",
        "backend": "rvc_onnx",
        "pitch_shift": float(pitch_shift),
        "warmth": 0.0,
        "presence": 0.0,
        "compression": 0.15,
        "output_gain_db": 0.0,
    }
    with open(target_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Registered AI voice '{name}' at: {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Register an RVC ONNX voice model bundle into Tvoice-s")
    parser.add_argument("--id", required=True, help="Unique identifier (e.g. anime_girl_01)")
    parser.add_argument("--name", required=True, help="Display name")
    parser.add_argument("--gender", default="female", choices=["female", "male", "custom"])
    parser.add_argument("--model", required=True, type=Path, help="Path to speaker model.onnx")
    parser.add_argument("--hubert", required=True, type=Path, help="Path to hubert.onnx")
    parser.add_argument("--rmvpe", required=True, type=Path, help="Path to rmvpe.onnx")
    parser.add_argument("--pitch", default=0.0, type=float, help="Default pitch shift in semitones")
    parser.add_argument("--desc", default="Imported AI Voice Model")

    args = parser.parse_args()
    register_voice(
        voice_id=args.id,
        name=args.name,
        gender=args.gender,
        model_path=args.model,
        hubert_path=args.hubert,
        rmvpe_path=args.rmvpe,
        pitch_shift=args.pitch,
        desc=args.desc,
    )


if __name__ == "__main__":
    main()
