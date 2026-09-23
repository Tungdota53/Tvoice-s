import argparse
import logging
import sys
import threading
import time
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
    parser.add_argument("--browser", action="store_true", help="Open in standard web browser instead of desktop window")
    args = parser.parse_args()

    logger.info("Initializing Aura Voice Engine...")
    pipeline = VoiceChangerPipeline()

    server = start_server(pipeline, host=args.host, port=args.port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://{args.host}:{args.port}"
    print(f"\n=======================================================")
    print(f"  AURA AI VOICE CHANGER - DESKTOP APP")
    print(f"  Local API endpoint: {url}")
    print(f"  Close desktop window to exit.")
    print(f"=======================================================\n")

    if args.browser:
        logger.info("Browser mode enabled. Opening browser...")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        try:
            import webview

            logger.info("Launching native desktop application window...")
            window = webview.create_window(
                title="Aura AI Voice Changer",
                url=url,
                width=1180,
                height=780,
                min_size=(960, 640),
                background_color="#0b0f19",
            )
            webview.start(debug=False)
        except Exception as e:
            logger.warning("Native window failed (%s), opening in browser fallback", e)
            webbrowser.open(url)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

    logger.info("Shutting down Aura Voice Engine...")
    pipeline.stop()
    server.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
