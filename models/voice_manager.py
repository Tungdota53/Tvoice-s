import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
import threading

from config import PATH_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    id: str
    name: str
    description: str = ""
    pitch_shift: float = 0.0  # Semitones (-12 to +12)
    gender: str = "custom"  # "male", "female", "robot", "custom"
    warmth: float = 0.0
    presence: float = 0.0
    compression: float = 0.35
    output_gain_db: float = 0.0
    hotkey: Optional[str] = None
    model_path: Optional[str] = None
    index_path: Optional[str] = None
    avatar_path: Optional[str] = None
    category: str = "ai_voice"
    backend: str = "rvc_onnx"
    model_ready: bool = False
    missing_files: tuple[str, ...] = ()


class VoiceManager:
    """Manages voice model discovery, loading, and instant hot-swapping."""

    def __init__(self, voices_dir: Path):
        self.voices_dir = Path(voices_dir)
        self.profiles: Dict[str, VoiceProfile] = {}
        self.active_voice_id: Optional[str] = None
        self._lock = threading.Lock()
        self.scan_voices()

    def scan_voices(self) -> List[VoiceProfile]:
        """Scan directory and reload all voice profiles."""
        with self._lock:
            self.profiles.clear()
            if not self.voices_dir.exists():
                self.voices_dir.mkdir(parents=True, exist_ok=True)

            # If empty and bundle has default voices, seed them
            bundled_voices = PATH_CONFIG.bundle_dir / "voices"
            if bundled_voices.exists() and bundled_voices.resolve() != self.voices_dir.resolve():
                existing_subdirs = [f for f in self.voices_dir.iterdir() if f.is_dir()]
                if not existing_subdirs:
                    logger.info("Seeding default voice profiles from bundle: %s", bundled_voices)
                    for item in bundled_voices.iterdir():
                        if item.is_dir():
                            dst = self.voices_dir / item.name
                            if not dst.exists():
                                shutil.copytree(item, dst)

            for folder in self.voices_dir.iterdir():
                if not folder.is_dir():
                    continue

                voice_id = folder.name
                config_file = folder / "config.json"
                model_file = folder / "model.onnx"
                index_file = folder / "index.faiss"
                avatar_file = folder / "avatar.png"

                name = voice_id.replace("_", " ").title()
                desc = "Custom Voice Model"
                pitch_shift = 0.0
                gender = "custom"
                warmth = 0.0
                presence = 0.0
                compression = 0.35
                output_gain_db = 0.0
                hotkey = None
                category = "ai_voice"
                backend = "rvc_onnx"

                if config_file.exists():
                    try:
                        with open(config_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            name = data.get("name", name)
                            desc = data.get("description", desc)
                            pitch_shift = float(data.get("pitch_shift", 0.0))
                            gender = data.get("gender", "custom")
                            warmth = float(data.get("warmth", 0.0))
                            presence = float(data.get("presence", 0.0))
                            compression = float(data.get("compression", 0.35))
                            output_gain_db = float(data.get("output_gain_db", 0.0))
                            hotkey = data.get("hotkey", None)
                            category = data.get("category", "ai_voice")
                            backend = data.get("backend", "rvc_onnx")
                    except Exception as e:
                        logger.error("Failed to parse %s: %s", config_file, e)

                required_files = ["model.onnx"]
                if backend == "rvc_onnx":
                    required_files.extend(["hubert.onnx", "rmvpe.onnx"])
                missing_files = tuple(name for name in required_files if not (folder / name).exists())

                profile = VoiceProfile(
                    id=voice_id,
                    name=name,
                    description=desc,
                    pitch_shift=pitch_shift,
                    gender=gender,
                    warmth=warmth,
                    presence=presence,
                    compression=compression,
                    output_gain_db=output_gain_db,
                    hotkey=hotkey,
                    model_path=str(model_file) if model_file.exists() else None,
                    index_path=str(index_file) if index_file.exists() else None,
                    avatar_path=str(avatar_file) if avatar_file.exists() else None,
                    category=category,
                    backend=backend,
                    model_ready=not missing_files,
                    missing_files=missing_files,
                )
                self.profiles[voice_id] = profile

            if self.profiles and self.active_voice_id not in self.profiles:
                self.active_voice_id = next(iter(self.profiles.keys()))

            logger.info("Loaded %d voice profiles", len(self.profiles))
            return list(self.profiles.values())

    def get_active_profile(self) -> Optional[VoiceProfile]:
        """Return profile of the currently active voice."""
        with self._lock:
            if self.active_voice_id and self.active_voice_id in self.profiles:
                return self.profiles[self.active_voice_id]
            return None

    def switch_voice(self, voice_id: str) -> bool:
        """Hot-swap active voice target. Non-blocking pointer swap."""
        with self._lock:
            if voice_id in self.profiles:
                self.active_voice_id = voice_id
                logger.info("Switched active voice to: %s", voice_id)
                return True
            logger.warning("Voice ID not found: %s", voice_id)
            return False

    def list_profiles_dict(self) -> List[dict]:
        """Return serialized list of all profiles for GUI/IPC."""
        with self._lock:
            return [asdict(p) for p in self.profiles.values()]
