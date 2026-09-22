import math

import numpy as np


class AudioQualityProcessor:
    """Stateful low-latency speech cleanup, tone shaping, compression, and limiting."""

    def __init__(self, sample_rate: int = 48_000):
        self.sample_rate = sample_rate
        self._dc_x1 = 0.0
        self._dc_y1 = 0.0
        self._low_state = 0.0
        self._presence_state = 0.0
        self._compressor_gain = 1.0

    def reset(self) -> None:
        self._dc_x1 = 0.0
        self._dc_y1 = 0.0
        self._low_state = 0.0
        self._presence_state = 0.0
        self._compressor_gain = 1.0

    def process(
        self,
        audio: np.ndarray,
        warmth: float = 0.0,
        presence: float = 0.0,
        compression: float = 0.35,
        output_gain_db: float = 0.0,
    ) -> np.ndarray:
        """Process one mono chunk without changing length or allocating large FFT buffers."""
        source = np.asarray(audio, dtype=np.float32)
        if source.size == 0:
            return source

        cleaned = self._remove_dc(source)
        shaped = self._shape_tone(cleaned, warmth, presence)
        compressed = self._compress(shaped, compression)

        gain = 10.0 ** (float(np.clip(output_gain_db, -12.0, 6.0)) / 20.0)
        # Smooth soft limiter. Prevents hard clipping and broken output at high gain.
        limited = np.tanh(compressed * gain * 1.15) / math.tanh(1.15)
        return np.clip(limited, -0.98, 0.98).astype(np.float32, copy=False)

    def _remove_dc(self, audio: np.ndarray) -> np.ndarray:
        output = np.empty_like(audio)
        x1 = self._dc_x1
        y1 = self._dc_y1
        coefficient = 0.995
        for index, sample in enumerate(audio):
            value = float(sample) - x1 + coefficient * y1
            output[index] = value
            x1 = float(sample)
            y1 = value
        self._dc_x1 = x1
        self._dc_y1 = y1
        return output

    def _shape_tone(self, audio: np.ndarray, warmth: float, presence: float) -> np.ndarray:
        warmth = float(np.clip(warmth, -1.0, 1.0))
        presence = float(np.clip(presence, -1.0, 1.0))
        output = np.empty_like(audio)

        low_alpha = 1.0 - math.exp(-2.0 * math.pi * 220.0 / self.sample_rate)
        presence_alpha = 1.0 - math.exp(-2.0 * math.pi * 2_800.0 / self.sample_rate)
        low_state = self._low_state
        presence_state = self._presence_state

        for index, sample in enumerate(audio):
            value = float(sample)
            low_state += low_alpha * (value - low_state)
            presence_state += presence_alpha * (value - presence_state)
            high_band = value - presence_state
            output[index] = value + warmth * low_state * 0.45 + presence * high_band * 0.35

        self._low_state = low_state
        self._presence_state = presence_state
        return output

    def _compress(self, audio: np.ndarray, amount: float) -> np.ndarray:
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount <= 0.001:
            return audio

        threshold = 10.0 ** ((-8.0 - amount * 12.0) / 20.0)
        ratio = 1.0 + amount * 5.0
        attack = math.exp(-1.0 / (0.004 * self.sample_rate))
        release = math.exp(-1.0 / (0.080 * self.sample_rate))
        gain = self._compressor_gain
        output = np.empty_like(audio)

        for index, sample in enumerate(audio):
            level = abs(float(sample)) + 1e-9
            target = 1.0
            if level > threshold:
                compressed_level = threshold + (level - threshold) / ratio
                target = compressed_level / level
            coefficient = attack if target < gain else release
            gain = coefficient * gain + (1.0 - coefficient) * target
            output[index] = float(sample) * gain

        self._compressor_gain = gain
        return output
