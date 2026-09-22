import numpy as np
import time

from config import AUDIO_CONFIG
from core.audio_quality import AudioQualityProcessor
from core.crossfade import CrossfadeEngine
from core.noise_gate import NoiseGate
from core.pipeline import VoiceChangerPipeline
from core.ring_buffer import RingBuffer


def test_ring_buffer():
    rb = RingBuffer(capacity=100)
    data = np.arange(50, dtype=np.float32)
    rb.write(data)
    assert rb.available_read() == 50

    read_data = rb.read(50)
    np.testing.assert_array_equal(data, read_data)
    assert rb.available_read() == 0
    print("[PASS] RingBuffer basic read/write test")


def test_crossfade():
    cf = CrossfadeEngine(crossfade_samples=32)
    chunk1 = np.ones(128, dtype=np.float32)
    chunk2 = np.ones(128, dtype=np.float32) * 2.0

    out1 = cf.process(chunk1)
    out2 = cf.process(chunk2)
    assert len(out1) == 128
    assert len(out2) == 128
    print("[PASS] Crossfade Overlap-Add test")


def test_noise_gate():
    gate = NoiseGate(threshold_db=-30.0, sample_rate=44100)

    # Silence
    silent = np.zeros(256, dtype=np.float32)
    out_silent, is_speaking = gate.process(silent)
    assert not is_speaking

    # Loud tone
    t = np.linspace(0, 0.01, 256)
    loud = (np.sin(2 * np.pi * 440 * t) * 0.9).astype(np.float32)
    out_loud, is_speaking_loud = gate.process(loud)
    assert is_speaking_loud
    print("[PASS] Noise Gate VAD test")


def test_audio_quality():
    processor = AudioQualityProcessor(AUDIO_CONFIG.sample_rate)
    t = np.arange(AUDIO_CONFIG.chunk_size) / AUDIO_CONFIG.sample_rate
    audio = (1.4 * np.sin(2 * np.pi * 300 * t) + 0.1).astype(np.float32)
    output = processor.process(audio, warmth=0.5, presence=0.4, compression=0.6, output_gain_db=3.0)
    assert len(output) == len(audio)
    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output)) <= 0.981
    print("[PASS] Audio quality and anti-clipping test")

def test_processing_latency():
    processor = AudioQualityProcessor(AUDIO_CONFIG.sample_rate)
    rng = np.random.default_rng(42)
    chunk = rng.normal(0.0, 0.2, AUDIO_CONFIG.chunk_size).astype(np.float32)
    iterations = 500
    start = time.perf_counter()
    for _ in range(iterations):
        processor.process(chunk, warmth=0.4, presence=0.4, compression=0.5)
    average_ms = (time.perf_counter() - start) * 1000.0 / iterations
    assert average_ms < AUDIO_CONFIG.block_time_ms
    print(f"[PASS] DSP average latency: {average_ms:.3f} ms")


def test_pipeline_status():
    pipeline = VoiceChangerPipeline()
    status = pipeline.get_status()
    assert "latency_ms" in status
    assert "active_voice_id" in status
    assert len(pipeline.voice_manager.profiles) >= 8
    print(f"[PASS] Pipeline initialized with {len(pipeline.voice_manager.profiles)} voices")


if __name__ == "__main__":
    test_ring_buffer()
    test_crossfade()
    test_noise_gate()
    test_audio_quality()
    test_processing_latency()
    test_pipeline_status()
    print("\nALL CORE ENGINE TESTS PASSED!")
