import logging
from typing import Any, Callable, Dict, List, Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from config import AUDIO_CONFIG
from core.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class AudioDeviceManager:
    """Manages audio devices query and stream lifecycle."""

    @staticmethod
    def list_devices() -> Dict[str, List[dict]]:
        """Return list of available input and output devices."""
        if sd is None:
            return {"input": [], "output": []}

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        input_devs = []
        output_devs = []

        for idx, dev in enumerate(devices):
            api_name = hostapis[dev["hostapi"]]["name"] if "hostapi" in dev else "Unknown"
            info = {
                "id": idx,
                "name": dev["name"],
                "host_api": api_name,
                "max_input_channels": dev["max_input_channels"],
                "max_output_channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
            }
            if dev["max_input_channels"] > 0:
                input_devs.append(info)
            if dev["max_output_channels"] > 0:
                output_devs.append(info)

        return {"input": input_devs, "output": output_devs}


class AudioIOEngine:
    """Real-time duplex audio streaming engine using sounddevice."""

    def __init__(
        self,
        sample_rate: int = AUDIO_CONFIG.sample_rate,
        chunk_size: int = AUDIO_CONFIG.chunk_size,
        channels: int = AUDIO_CONFIG.channels,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        monitor_device: Optional[int] = None,
        process_callback: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.input_device = input_device
        self.output_device = output_device
        self.monitor_device = monitor_device
        self.process_callback = process_callback

        # State flags
        self.is_running = False
        self.bypass = False
        self.is_muted = False
        self.input_gain = 1.0
        self.output_gain = 1.0
        self.monitor_gain = 1.0

        # Ring buffers (buffer 1 second of audio)
        buffer_capacity = int(sample_rate * 1.5)
        self.in_buffer = RingBuffer(buffer_capacity, channels=channels)
        self.out_buffer = RingBuffer(buffer_capacity, channels=channels)
        self.monitor_buffer = RingBuffer(buffer_capacity, channels=channels)

        self._stream: Optional[Any] = None
        self._monitor_stream: Optional[Any] = None

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """Full-duplex stream callback."""
        if status:
            logger.warning("Audio stream status: %s", status)

        input_audio = indata[:, 0].copy() if self.channels == 1 else indata.copy()

        # Apply input gain
        if self.input_gain != 1.0:
            input_audio = input_audio * self.input_gain

        if self.is_muted:
            outdata.fill(0)
            return

        if self.bypass or self.process_callback is None:
            processed_audio = input_audio
        else:
            try:
                processed_audio = self.process_callback(input_audio)
            except Exception as ex:
                logger.error("Error in process_callback: %s", ex)
                processed_audio = input_audio

        # Apply output gain
        if self.output_gain != 1.0:
            processed_audio = processed_audio * self.output_gain
        processed_audio = np.nan_to_num(processed_audio, nan=0.0, posinf=0.98, neginf=-0.98)
        processed_audio = np.clip(processed_audio, -0.98, 0.98)

        # Populate primary output
        if self.channels == 1:
            outdata[:, 0] = processed_audio
            if outdata.shape[1] > 1:
                outdata[:, 1:] = 0
        else:
            outdata[:] = processed_audio

        # Feed monitor buffer if monitoring is enabled
        if self._monitor_stream is not None and self._monitor_stream.active:
            self.monitor_buffer.write(processed_audio * self.monitor_gain)

    def _monitor_callback(self, outdata, frames, time_info, status):
        """Callback for secondary headphone monitoring."""
        data = self.monitor_buffer.read(frames)
        if self.channels == 1:
            outdata[:, 0] = data
            if outdata.shape[1] > 1:
                outdata[:, 1:] = 0
        else:
            outdata[:] = data

    def start(self):
        """Start audio streams."""
        if self.is_running or sd is None:
            return

        self.in_buffer.clear()
        self.out_buffer.clear()
        self.monitor_buffer.clear()

        # Primary duplex stream (Mic In -> App/Cable Out)
        self._stream = sd.Stream(
            device=(self.input_device, self.output_device),
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype="float32",
            channels=self.channels,
            latency="low",
            callback=self._audio_callback,
        )
        self._stream.start()

        # Optional monitor stream
        if self.monitor_device is not None:
            try:
                self._monitor_stream = sd.OutputStream(
                    device=self.monitor_device,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    dtype="float32",
                    channels=self.channels,
                    latency="low",
                    callback=self._monitor_callback,
                )
                self._monitor_stream.start()
            except Exception as e:
                logger.warning("Could not start monitor stream: %s", e)

        self.is_running = True
        logger.info("Audio engine started.")

    def stop(self):
        """Stop audio streams."""
        if not self.is_running:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._monitor_stream is not None:
            self._monitor_stream.stop()
            self._monitor_stream.close()
            self._monitor_stream = None

        self.is_running = False
        logger.info("Audio engine stopped.")
