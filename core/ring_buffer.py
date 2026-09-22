import threading
import numpy as np


class RingBuffer:
    """Thread-safe circular ring buffer for real-time audio sample streaming."""

    def __init__(self, capacity: int, channels: int = 1, dtype: str = "float32"):
        self.capacity = int(capacity)
        self.channels = int(channels)
        self.dtype = np.dtype(dtype)

        if self.channels == 1:
            self.buffer = np.zeros(self.capacity, dtype=self.dtype)
        else:
            self.buffer = np.zeros((self.capacity, self.channels), dtype=self.dtype)

        self._read_ptr = 0
        self._write_ptr = 0
        self._size = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        """Write array of samples into ring buffer. Returns number of samples written."""
        data = np.asarray(data, dtype=self.dtype)
        num_samples = len(data)

        with self._lock:
            available_space = self.capacity - self._size
            if num_samples > available_space:
                # Buffer overflow: discard oldest samples
                drop_count = num_samples - available_space
                self._read_ptr = (self._read_ptr + drop_count) % self.capacity
                self._size -= drop_count

            # Write in 1 or 2 chunks (handle wrap-around)
            first_chunk = min(num_samples, self.capacity - self._write_ptr)
            second_chunk = num_samples - first_chunk

            self.buffer[self._write_ptr : self._write_ptr + first_chunk] = data[:first_chunk]
            if second_chunk > 0:
                self.buffer[:second_chunk] = data[first_chunk:]

            self._write_ptr = (self._write_ptr + num_samples) % self.capacity
            self._size += num_samples

            return num_samples

    def read(self, num_samples: int) -> np.ndarray:
        """Read requested number of samples. Pad with zeros if underrun."""
        with self._lock:
            samples_to_read = min(num_samples, self._size)

            if self.channels == 1:
                out = np.zeros(num_samples, dtype=self.dtype)
            else:
                out = np.zeros((num_samples, self.channels), dtype=self.dtype)

            if samples_to_read > 0:
                first_chunk = min(samples_to_read, self.capacity - self._read_ptr)
                second_chunk = samples_to_read - first_chunk

                out[:first_chunk] = self.buffer[self._read_ptr : self._read_ptr + first_chunk]
                if second_chunk > 0:
                    out[first_chunk:samples_to_read] = self.buffer[:second_chunk]

                self._read_ptr = (self._read_ptr + samples_to_read) % self.capacity
                self._size -= samples_to_read

            return out

    def available_read(self) -> int:
        """Return number of samples ready to read."""
        with self._lock:
            return self._size

    def available_write(self) -> int:
        """Return remaining empty sample slots."""
        with self._lock:
            return self.capacity - self._size

    def clear(self) -> None:
        """Reset buffer state."""
        with self._lock:
            self._read_ptr = 0
            self._write_ptr = 0
            self._size = 0
            self.buffer.fill(0)
