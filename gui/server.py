import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from config import PATH_CONFIG
from core.audio_io import AudioDeviceManager
from core.pipeline import VoiceChangerPipeline

logger = logging.getLogger(__name__)

# Search in relative directory or inside bundled PyInstaller assets
WEB_DIR = Path(__file__).resolve().parent / "web"
if not WEB_DIR.exists() and (PATH_CONFIG.bundle_dir / "gui" / "web").exists():
    WEB_DIR = PATH_CONFIG.bundle_dir / "gui" / "web"


class VoiceAPIHandler(BaseHTTPRequestHandler):
    pipeline: VoiceChangerPipeline = None  # Injected before server starts

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            return {}
        raw = self.rfile.read(content_len).decode("utf-8")
        return json.loads(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            html_path = WEB_DIR / "index.html"
            if html_path.exists():
                content = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "UI Not Found")
            return

        if path == "/api/devices":
            devices = AudioDeviceManager.list_devices()
            self._send_json(devices)
            return

        if path == "/api/voices":
            voices = self.pipeline.voice_manager.list_profiles_dict()
            self._send_json(voices)
            return

        if path == "/api/status":
            status = self.pipeline.get_status()
            self._send_json(status)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_json()

        if path == "/api/switch_voice":
            voice_id = body.get("voice_id", "")
            success = self.pipeline.switch_voice(voice_id)
            self._send_json({"success": success, "active": voice_id})
            return

        if path == "/api/rescan_voices":
            profiles = self.pipeline.voice_manager.scan_voices()
            self._send_json({"count": len(profiles), "voices": self.pipeline.voice_manager.list_profiles_dict()})
            return

        if path == "/api/import_voice":
            voice_id = body.get("voice_id", "").strip().lower().replace(" ", "_")
            name = body.get("name", "").strip() or voice_id.title()
            gender = body.get("gender", "custom")
            pitch_shift = float(body.get("pitch_shift", 0.0))

            if not voice_id:
                self._send_json({"success": False, "error": "voice_id is required"}, status=400)
                return

            voice_dir = self.pipeline.voice_manager.voices_dir / voice_id
            voice_dir.mkdir(parents=True, exist_ok=True)

            config_data = {
                "name": name,
                "description": body.get("description", "Custom imported AI voice"),
                "gender": gender,
                "category": "ai_voice",
                "backend": "rvc_onnx",
                "pitch_shift": pitch_shift,
                "warmth": 0.0,
                "presence": 0.0,
                "compression": 0.15,
                "output_gain_db": 0.0,
            }

            config_file = voice_dir / "config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            self.pipeline.voice_manager.scan_voices()
            self._send_json({
                "success": True,
                "voice_id": voice_id,
                "folder": str(voice_dir),
                "required_files": ["model.onnx", "hubert.onnx", "rmvpe.onnx"],
            })
            return

        if path == "/api/toggle_engine":
            if self.pipeline.audio_io.is_running:
                self.pipeline.stop()
            else:
                self.pipeline.start()
            self._send_json({"running": self.pipeline.audio_io.is_running})
            return

        if path == "/api/set_pitch":
            pitch = float(body.get("pitch", 0.0))
            self.pipeline.set_pitch_offset(pitch)
            self._send_json({"pitch": pitch})
            return

        if path == "/api/set_denoise":
            threshold_db = float(body.get("threshold_db", -45.0))
            self.pipeline.set_denoise_threshold(threshold_db)
            self._send_json({"threshold_db": threshold_db})
            return

        if path == "/api/set_volume":
            vol_type = body.get("type", "")
            gain = float(body.get("gain", 1.0))
            if vol_type == "input":
                self.pipeline.audio_io.input_gain = gain
            elif vol_type == "output":
                self.pipeline.audio_io.output_gain = gain
            elif vol_type == "monitor":
                self.pipeline.audio_io.monitor_gain = gain
            self._send_json({"type": vol_type, "gain": gain})
            return

        if path == "/api/set_devices":
            in_dev = body.get("input_device")
            out_dev = body.get("output_device")
            mon_dev = body.get("monitor_device")

            was_running = self.pipeline.audio_io.is_running
            if was_running:
                self.pipeline.stop()

            self.pipeline.audio_io.input_device = in_dev
            self.pipeline.audio_io.output_device = out_dev
            self.pipeline.audio_io.monitor_device = mon_dev

            if was_running:
                self.pipeline.start()

            self._send_json({"success": True, "restarted": was_running})
            return

        self.send_error(404, "Unknown endpoint")


def start_server(pipeline: VoiceChangerPipeline, host: str = "127.0.0.1", port: int = 7860):
    VoiceAPIHandler.pipeline = pipeline
    server = ThreadingHTTPServer((host, port), VoiceAPIHandler)
    logger.info("UI Server running at: http://%s:%d", host, port)
    return server
