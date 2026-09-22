import numpy as np


class CrossfadeEngine:
    """Seamless Overlap-Add (OLA) crossfade processor to eliminate chunk boundary clicks."""

    def __init__(self, crossfade_samples: int = 128, window_type: str = "cosine"):
        self.crossfade_samples = int(crossfade_samples)
        self.window_type = window_type
        self.prev_tail: np.ndarray = np.zeros(self.crossfade_samples, dtype=np.float32)

        # Precompute fade curves
        if window_type == "cosine":
            # Cosine fade: sum of power is 1
            t = np.linspace(0, np.pi / 2, self.crossfade_samples, endpoint=False)
            self.fade_in = np.sin(t).astype(np.float32)
            self.fade_out = np.cos(t).astype(np.float32)
        elif window_type == "hann":
            t = np.linspace(0, np.pi, self.crossfade_samples, endpoint=False)
            hann = 0.5 * (1 - np.cos(t))
            self.fade_in = hann.astype(np.float32)
            self.fade_out = (1.0 - hann).astype(np.float32)
        else:  # Linear
            self.fade_in = np.linspace(0.0, 1.0, self.crossfade_samples, endpoint=False, dtype=np.float32)
            self.fade_out = np.linspace(1.0, 0.0, self.crossfade_samples, endpoint=False, dtype=np.float32)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Apply crossfade on the input chunk using the cached tail of the previous chunk.

        Returns seamlessly stitched chunk of the same length.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if len(chunk) < self.crossfade_samples:
            return chunk

        out = chunk.copy()

        # Blend previous tail with current head
        head = out[: self.crossfade_samples]
        blended = (self.prev_tail * self.fade_out) + (head * self.fade_in)
        out[: self.crossfade_samples] = blended

        # Cache current chunk tail for the next frame
        self.prev_tail = chunk[-self.crossfade_samples :].copy()

        return out

    def reset(self) -> None:
        """Reset internal tail buffer."""
        self.prev_tail.fill(0)
