import logging
import time
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from core.audio_quality import AudioQualityProcessor
from models.rvc_pipeline import RVCInferencePipeline, RVCModelBundle
from models.voice_manager import VoiceProfile

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages AI voice conversion with proper multi-tensor RVC ONNX routing."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.quality = AudioQualityProcessor(sample_rate=sample_rate)
        self.session = None
        self.rvc_pipeline: RVCInferencePipeline = RVCInferencePipeline()
        self.current_model_path: Optional[str] = None
        self.providers = self._detect_providers()
        self.rvc_pipeline.providers = self.providers
        self.last_latency_ms: float = 0.0
        self.backend_status: str = "model_missing"
        self.last_error: Optional[str] = None

    def _detect_providers(self) -> list[str]:
        """Detect fastest available ONNX Execution Providers."""
        if ort is None:
            return []
        available = ort.get_available_providers()
        chosen = []
        if "CUDAExecutionProvider" in available:
            chosen.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            chosen.append("DmlExecutionProvider")
        chosen.append("CPUExecutionProvider")
        logger.info("ONNX Execution Providers: %s", chosen)
        return chosen

    def load_model(self, model_path: Optional[str], profile: Optional[VoiceProfile] = None) -> bool:
        """Load or switch voice model session with backend-aware validation."""
        if not model_path or ort is None:
            self.session = None
            self.rvc_pipeline.unload()
            self.current_model_path = None
            self.backend_status = "runtime_missing" if model_path and ort is None else "model_missing"
            return False

        if profile and profile.backend == "rvc_onnx":
            voice_dir = Path(model_path).parent
            bundle = RVCModelBundle(
                model_path=Path(model_path),
                hubert_path=voice_dir / "hubert.onnx",
                rmvpe_path=voice_dir / "rmvpe.onnx",
            )
            success = self.rvc_pipeline.load_bundle(bundle)
            if success:
                self.current_model_path = model_path
                self.backend_status = "ready"
                self.last_error = None
                return True
            else:
                self.current_model_path = None
                self.backend_status = "model_missing"
                self.last_error = self.rvc_pipeline.last_error
                return False

        # Fallback for generic legacy models
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=self.providers)
            self.current_model_path = model_path
            self.backend_status = "ready"
            self.last_error = None
            logger.info("Loaded generic ONNX model: %s", model_path)
            return True
        except Exception as e:
            logger.error("Failed to load generic ONNX model %s: %s", model_path, e)
            self.session = None
            self.current_model_path = None
            self.backend_status = "load_error"
            self.last_error = str(e)
            return False

    def process_chunk(self, chunk: np.ndarray, profile: Optional[VoiceProfile]) -> np.ndarray:
        """Process incoming audio chunk with verified AI voice conversion."""
        t_start = time.perf_counter()

        processed = np.asarray(chunk, dtype=np.float32)

        # Path A: Genuine RVC pipeline if active
        if self.rvc_pipeline.is_ready:
            pitch = profile.pitch_shift if profile else 0.0
            processed = self.rvc_pipeline.convert_voice(processed, pitch_shift=pitch)
        # Path B: Legacy generic ONNX session
        elif self.session is not None:
            try:
                inp_name = self.session.get_inputs()[0].name
                inp_tensor = processed[np.newaxis, np.newaxis, :].astype(np.float32)
                outputs = self.session.run(None, {inp_name: inp_tensor})
                model_out = outputs[0].squeeze()
                processed = model_out.astype(np.float32)
            except Exception as e:
                logger.error("ONNX inference failed: %s", e)

        # Output limiter to prevent digital clipping
        output = self.quality.process(processed, compression=0.15)

        t_end = time.perf_counter()
        self.last_latency_ms = (t_end - t_start) * 1000.0

        return output
