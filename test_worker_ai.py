import time
import numpy as np

from core.worker_pipeline import AudioWorkerPipeline
from models.rvc_pipeline import RVCInferencePipeline, RVCModelBundle


def test_audio_worker_isolation():
    chunk_size = 480

    def slow_ai_process(chunk):
        # Simulate an inference delay of 5ms
        time.sleep(0.005)
        return chunk * 0.9

    worker = AudioWorkerPipeline(chunk_size=chunk_size, process_fn=slow_ai_process, max_queue_chunks=3)
    worker.start()

    # Audio callback simulates rapid feeding
    for _ in range(10):
        dummy_in = np.ones(chunk_size, dtype=np.float32)
        worker.push_input(dummy_in)
        out = worker.pull_output(fallback=dummy_in)
        assert len(out) == chunk_size
        assert np.all(np.isfinite(out))

    time.sleep(0.05)
    stats = worker.get_stats()
    worker.stop()

    assert stats["worker_inference_ms"] > 0.0
    print(f"[PASS] AudioWorkerPipeline test passed: inference {stats['worker_inference_ms']}ms, drops {stats['drop_count']}")


def test_rvc_pipeline_bundle_guard():
    pipeline = RVCInferencePipeline()
    # Missing bundle verification
    bundle = RVCModelBundle(
        model_path=np.testing.__file__,  # arbitrary file
        hubert_path=np.testing.__file__,
        rmvpe_path=np.testing.__file__,
    )
    # Shouldn't crash and should safely fail validation
    success = pipeline.load_bundle(bundle)
    assert not success
    assert pipeline.last_error is not None
    print(f"[PASS] RVC pipeline bundle guard passed (reported: {pipeline.last_error})")


if __name__ == "__main__":
    test_audio_worker_isolation()
    test_rvc_pipeline_bundle_guard()
    print("ALL WORKER & AI BACKEND TESTS PASSED!")
