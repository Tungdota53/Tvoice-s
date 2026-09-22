# Tvoice-s — Real-time Voice Changer

Ứng dụng đổi màu giọng thời gian thực cho Windows, dùng audio 48 kHz, block 10 ms,
noise gate, tone shaping, compressor và soft limiter chống vỡ tiếng.

## Chạy ứng dụng

1. Cài Python 3.11 hoặc 3.12 để có tương thích ONNX Runtime tốt nhất.
2. Tạo môi trường và cài dependency: `uv venv` rồi `uv pip install -r requirements.txt`.
3. Chạy `run.bat` hoặc `.venv\Scripts\python.exe main.py`.
4. Chọn microphone, VB-CABLE output và tai nghe monitor trong giao diện.

## Preset

Dự án kèm 8 preset tone. Preset chỉ đổi EQ/compression/màu âm. Muốn chuyển sang
nhân vật hoặc người nói khác với chất lượng cao, đặt model ONNX tương thích và có
quyền sử dụng vào `voices/<voice-id>/model.onnx`.

Không dùng giọng người thật khi chưa có sự đồng ý. Không mạo danh hoặc lừa đảo.

## Kiểm thử

Chạy `.venv\Scripts\python.exe test_engine.py`.