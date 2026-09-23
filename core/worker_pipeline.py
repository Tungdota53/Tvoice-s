import logging
import queue
import threading
import time
from typing import Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)


class AudioWorkerPipeline:
    """Decouples real-time audio I/O callbacks from heavy AI inference tasks.

    Uses bounded queues with drop-oldest semantics on overflow and zero/pass-through
    fallback on underrun to keep the audio callback bounded and stutter-free.
    """

    def __init__(
        self,
        chunk_size: int,
        process_fn: Callable[[np.ndarray], np.ndarray],
        max_queue_chunks: int = 4,
    ):
        self.chunk_size = int(chunk_size)
        self.process_fn = process_fn
        self.max_queue_chunks = max(2, int(max_queue_chunks))

        self._in_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_chunks)
        self._out_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_chunks)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Telemetry metrics
        self.underrun_count: int = 0
        self.drop_count: int = 0
        self.last_inference_ms: float = 0.0
        self.last_queue_latency_ms: float = 0.0
        self._lock = threading.Lock()

    def start(self):
        """Start background processing worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="AudioInferenceWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info("Audio worker thread started (max_chunks=%d)", self.max_queue_chunks)

    def stop(self):
        """Stop worker thread and wait for completion."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.clear()
        logger.info("Audio worker thread stopped")

    def clear(self):
        """Drain both input and output queues."""
        while not self._in_queue.empty():
            try:
                self._in_queue.get_nowait()
            except queue.Empty:
                break
        while not self._out_queue.empty():
            try:
                self._out_queue.get_nowait()
            except queue.Empty:
                break

    def push_input(self, chunk: np.ndarray) -> None:
        """Non-blocking push from audio callback to worker."""
        arrival_time = time.perf_counter()
        item = (arrival_time, np.asarray(chunk, dtype=np.float32).copy())

        try:
            self._in_queue.put_nowait(item)
        except queue.Full:
            # Drop oldest frame to prioritize real-time fresh voice
            try:
                self._in_queue.get_nowait()
                with self._lock:
                    self.drop_count += 1
            except queue.Empty:
                pass
            try:
                self._in_queue.put_nowait(item)
            except queue.Full:
                pass

    def pull_output(self, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        """Non-blocking fetch for PortAudio output callback."""
        try:
            out_chunk = self._out_queue.get_nowait()
            return out_chunk
        except queue.Empty:
            with self._lock:
                self.underrun_count += 1
            if fallback is not None:
                return fallback
            return np.zeros(self.chunk_size, dtype=np.float32)

    def _worker_loop(self):
        """Worker thread executing heavy model inference."""
        while not self._stop_event.is_set():
            try:
                arrival_time, chunk = self._in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t_start = time.perf_counter()
            queue_latency = (t_start - arrival_time) * 1000.0

            try:
                processed = self.process_fn(chunk)
                processed = np.asarray(processed, dtype=np.float32)
                if len(processed) != self.chunk_size:
                    logger.warning(
                        "Worker output length mismatch: got %d, expected %d",
                        len(processed),
                        self.chunk_size,
                    )
            except Exception as e:
                logger.error("Audio worker processing error: %s", e)
                processed = chunk

            t_end = time.perf_counter()
            infer_ms = (t_end - t_start) * 1000.0

            with self._lock:
                self.last_inference_ms = infer_ms
                self.last_queue_latency_ms = queue_latency

            # Push processed frame to output queue
            try:
                self._out_queue.put_nowait(processed)
            except queue.Full:
                try:
                    self._out_queue.get_nowait()
                    with self._lock:
                        self.drop_count += 1
                except queue.Empty:
                    pass
                try:
                    self._out_queue.put_nowait(processed)
                except queue.Full:
                    pass

    def get_stats(self) -> dict:
        """Return real-time queue health telemetry."""
        with self._lock:
            return {
                "in_queue_size": self._in_queue.qsize(),
                "out_queue_size": self._out_queue.qsize(),
                "underrun_count": self.underrun_count,
                "drop_count": self.drop_count,
                "worker_inference_ms": round(self.last_inference_ms, 2),
                "queue_latency_ms": round(self.last_queue_latency_ms, 2),
            }
