import logging
from typing import Optional
import numpy as np

from config import AUDIO_CONFIG, PATH_CONFIG
from core.audio_io import AudioIOEngine
from core.noise_gate import NoiseGate
from models.inference_engine import InferenceEngine
from models.voice_manager import VoiceManager, VoiceProfile

logger = logging.getLogger(__name__)


class VoiceChangerPipeline:
    """Master pipeline orchestrating Audio I/O, VAD, Voice Management, and AI Inference."""

    def __init__(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        monitor_device: Optional[int] = None,
    ):
        self.voice_manager = VoiceManager(PATH_CONFIG.voices_dir)
        self.inference_engine = InferenceEngine(sample_rate=AUDIO_CONFIG.sample_rate)
        self.noise_gate = NoiseGate(
            threshold_db=-45.0,
            sample_rate=AUDIO_CONFIG.sample_rate,
        )

        self.audio_io = AudioIOEngine(
            sample_rate=AUDIO_CONFIG.sample_rate,
            chunk_size=AUDIO_CONFIG.chunk_size,
            channels=AUDIO_CONFIG.channels,
            input_device=input_device,
            output_device=output_device,
            monitor_device=monitor_device,
            process_callback=self._process_audio,
        )

        # Dynamic adjustments
        self.custom_pitch_offset: float = 0.0
        self.init_active_voice()

    def init_active_voice(self):
        """Pre-load first available voice profile model."""
        active = self.voice_manager.get_active_profile()
        if active and active.model_path:
            self.inference_engine.load_model(active.model_path, active)

    def _process_audio(self, indata: np.ndarray) -> np.ndarray:
        """Pipeline callback invoked per audio chunk."""
        # Step 1: Noise Gate / VAD
        gated_audio, is_speech = self.noise_gate.process(indata)

        if not is_speech:
            return gated_audio

        # Step 2: Retrieve Active Profile
        profile = self.voice_manager.get_active_profile()

        # Combine profile pitch with runtime user offset
        effective_profile = profile
        if profile and self.custom_pitch_offset != 0.0:
            effective_profile = VoiceProfile(
                id=profile.id,
                name=profile.name,
                pitch_shift=profile.pitch_shift + self.custom_pitch_offset,
                gender=profile.gender,
                warmth=profile.warmth,
                presence=profile.presence,
                compression=profile.compression,
                output_gain_db=profile.output_gain_db,
                model_path=profile.model_path,
            )

        # Step 3: Run Inference / Conversion
        out = self.inference_engine.process_chunk(gated_audio, effective_profile)
        return out

    def switch_voice(self, voice_id: str) -> bool:
        """Switch voice profile and reload model asynchronously."""
        success = self.voice_manager.switch_voice(voice_id)
        if success:
            profile = self.voice_manager.get_active_profile()
            if profile and profile.model_path:
                self.inference_engine.load_model(profile.model_path, profile)
            else:
                self.inference_engine.load_model(None)
        return success

    def set_pitch_offset(self, semitones: float):
        """Fine-tune pitch in semitones (-12 to +12)."""
        self.custom_pitch_offset = float(np.clip(semitones, -12.0, 12.0))

    def set_denoise_threshold(self, threshold_db: float):
        """Adjust noise gate threshold (-80 to 0 dBFS)."""
        self.noise_gate.threshold_db = float(threshold_db)

    def start(self):
        self.audio_io.start()

    def stop(self):
        self.audio_io.stop()

    def get_status(self) -> dict:
        """Return engine diagnostics for GUI monitoring."""
        active = self.voice_manager.get_active_profile()
        return {
            "running": self.audio_io.is_running,
            "bypass": self.audio_io.bypass,
            "muted": self.audio_io.is_muted,
            "is_speaking": self.noise_gate.is_speaking,
            "latency_ms": round(self.inference_engine.last_latency_ms, 2),
            "active_voice_id": active.id if active else None,
            "active_voice_name": active.name if active else None,
            "model_ready": active.model_ready if active else False,
            "backend": active.backend if active else None,
            "backend_status": self.inference_engine.backend_status,
            "backend_error": self.inference_engine.last_error,
            "pitch_offset": self.custom_pitch_offset,
            "noise_threshold_db": self.noise_gate.threshold_db,
            "input_gain": self.audio_io.input_gain,
            "output_gain": self.audio_io.output_gain,
            "monitor_gain": self.audio_io.monitor_gain,
        }
