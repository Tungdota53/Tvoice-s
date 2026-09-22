import argparse
import logging
import sys
import threading
import webbrowser

from core.pipeline import VoiceChangerPipeline
from gui.server import start_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AuraVoice")


def main():
    parser = argparse.ArgumentParser(description="Aura Real-Time AI Voice Changer")
    parser.add_argument("--host", default="127.0.0.1", help="Host address for UI server")
    parser.add_argument("--port", type=int, default=7860, help="Port for UI server")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    logger.info("Initializing Aura Voice Engine...")
    pipeline = VoiceChangerPipeline()

    server = start_server(pipeline, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"\n=======================================================")
    print(f"  AURA AI VOICE CHANGER - READY")
    print(f"  Open UI in browser: {url}")
    print(f"  Press Ctrl+C to terminate.")
    print(f"=======================================================\n")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Aura Voice Engine...")
        pipeline.stop()
        server.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
