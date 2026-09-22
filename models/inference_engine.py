import logging
import time
from typing import Optional
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from core.audio_quality import AudioQualityProcessor
from models.voice_manager import VoiceProfile

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages AI inference via ONNX Runtime with DSP fallback."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.quality = AudioQualityProcessor(sample_rate=sample_rate)
        self.session = None
        self.current_model_path: Optional[str] = None
        self.providers = self._detect_providers()
        self.last_latency_ms: float = 0.0

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

    def load_model(self, model_path: Optional[str]) -> bool:
        """Load or switch ONNX model session."""
        if model_path == self.current_model_path:
            return True

        if not model_path or ort is None:
            self.session = None
            self.current_model_path = None
            return False

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=self.providers)
            self.current_model_path = model_path
            logger.info("Loaded ONNX model: %s", model_path)
            return True
        except Exception as e:
            logger.error("Failed to load ONNX model %s: %s", model_path, e)
            self.session = None
            self.current_model_path = None
            return False

    def process_chunk(self, chunk: np.ndarray, profile: Optional[VoiceProfile]) -> np.ndarray:
        """Process incoming audio chunk with pitch shift and AI model inference."""
        t_start = time.perf_counter()

        processed = np.asarray(chunk, dtype=np.float32)

        # Step 2: If ONNX model session is active, run inference
        if self.session is not None:
            try:
                # Shape input tensor [1, 1, samples]
                inp_name = self.session.get_inputs()[0].name
                inp_tensor = processed[np.newaxis, np.newaxis, :].astype(np.float32)
                outputs = self.session.run(None, {inp_name: inp_tensor})
                model_out = outputs[0].squeeze()
                processed = model_out.astype(np.float32)
            except Exception as e:
                logger.error("ONNX inference failed: %s", e)

        output = self.quality.process(
            processed,
            warmth=profile.warmth if profile else 0.0,
            presence=profile.presence if profile else 0.0,
            compression=profile.compression if profile else 0.35,
            output_gain_db=profile.output_gain_db if profile else 0.0,
        )

        t_end = time.perf_counter()
        self.last_latency_ms = (t_end - t_start) * 1000.0

        return output
