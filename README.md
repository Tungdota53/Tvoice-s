# Tvoice-s — Real-time Voice Changer

Ứng dụng đổi màu giọng thời gian thực cho Windows, dùng audio 48 kHz, block 10 ms,
noise gate, tone shaping, compressor và soft limiter chống vỡ tiếng.

## Chạy ứng dụng

1. Cài Python 3.11 hoặc 3.12 để có tương thích ONNX Runtime tốt nhất.
2. Tạo môi trường và cài dependency: `uv venv` rồi `uv pip install -r requirements.txt`.
3. Chạy `run.bat` hoặc `.venv\Scripts\python.exe main.py`.
4. Chọn microphone, VB-CABLE output và tai nghe monitor trong giao diện.

## Giọng AI thật

Mỗi giọng RVC ONNX cần ba artifact tương thích trong `voices/<voice-id>/`:
`model.onnx`, `hubert.onnx`, `rmvpe.onnx`. Giao diện đánh dấu rõ model thiếu và
không gọi EQ/pitch là giọng AI. Pipeline RVC nhiều đầu vào chưa được cài sẽ báo
`backend_not_installed`, thay vì chạy model như waveform ONNX một đầu vào sai chuẩn.

Các thư mục preset cũ chỉ còn metadata/tham khảo; không tạo danh tính giọng mới.
Checkpoint phải được huấn luyện từ dữ liệu có đồng ý hoặc có giấy phép rõ ràng.

Không dùng giọng người thật khi chưa có sự đồng ý. Không mạo danh hoặc lừa đảo.

## Kiểm thử

Chạy `.venv\Scripts\python.exe test_engine.py`.