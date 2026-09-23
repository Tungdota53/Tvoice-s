import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    chunk_size: int = 480           # 10ms at 48kHz
    channels: int = 1               # Mono audio stream
    dtype: str = "float32"
    block_time_ms: float = 10.0     # Target latency per buffer block
    crossfade_samples: int = 96     # 2ms transition window


@dataclass
class PathConfig:
    if getattr(sys, "frozen", False):
        # Running inside packaged executable
        root_dir: Path = Path(sys.executable).resolve().parent
        bundle_dir: Path = Path(getattr(sys, "_MEIPASS", root_dir))
    else:
        root_dir: Path = Path(__file__).resolve().parent
        bundle_dir: Path = root_dir

    voices_dir: Path = root_dir / "voices"
    models_dir: Path = root_dir / "models"


AUDIO_CONFIG = AudioConfig()
PATH_CONFIG = PathConfig()
