import numpy as np


class NoiseGate:
    """Dynamic Noise Gate and Voice Activity Detector with attack/hold/release envelope."""

    def __init__(
        self,
        threshold_db: float = -45.0,
        attack_ms: float = 10.0,
        hold_ms: float = 80.0,
        release_ms: float = 100.0,
        sample_rate: int = 44100,
    ):
        self.threshold_db = threshold_db
        self.sample_rate = sample_rate

        # Convert times to frame counts
        self.attack_frames = max(1, int((attack_ms / 1000.0) * sample_rate))
        self.hold_frames = max(1, int((hold_ms / 1000.0) * sample_rate))
        self.release_frames = max(1, int((release_ms / 1000.0) * sample_rate))

        self.current_gain: float = 0.0
        self.hold_counter: int = 0
        self.is_speaking: bool = False

    def _rms_db(self, chunk: np.ndarray) -> float:
        """Calculate Root Mean Square energy in dBFS."""
        square_mean = np.mean(chunk**2)
        if square_mean <= 1e-12:
            return -120.0
        return 20.0 * np.log10(np.sqrt(square_mean))

    def process(self, chunk: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply noise gate to chunk.

        Returns:
            processed_chunk (np.ndarray): Audio after gating.
            is_active (bool): Whether voice activity is detected.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        level_db = self._rms_db(chunk)
        chunk_len = len(chunk)

        is_above_threshold = level_db > self.threshold_db

        if is_above_threshold:
            self.is_speaking = True
            self.hold_counter = self.hold_frames
            target_gain = 1.0
        else:
            if self.hold_counter > 0:
                self.hold_counter -= chunk_len
                target_gain = 1.0
                self.is_speaking = True
            else:
                target_gain = 0.0
                self.is_speaking = False

        # Linear envelope smoothing
        if target_gain > self.current_gain:
            step = chunk_len / self.attack_frames
            self.current_gain = min(1.0, self.current_gain + step)
        else:
            step = chunk_len / self.release_frames
            self.current_gain = max(0.0, self.current_gain - step)

        processed = chunk * self.current_gain
        return processed, self.is_speaking
