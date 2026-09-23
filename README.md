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

## Đóng gói ứng dụng chạy Windows (.exe)

Ứng dụng hỗ trợ đóng gói độc lập để chạy trên máy không cần cài đặt Python:

1. Chạy file `build.bat` hoặc lệnh:
   ```cmd
   .venv\Scripts\pyinstaller.exe --noconfirm Tvoice.spec
   ```
2. Thư mục phát hành được tạo tại `dist\Tvoice-s\`.
3. Chạy `dist\Tvoice-s\Tvoice-s.exe` để mở ứng dụng.
4. Thư mục `dist\Tvoice-s\voices\` nằm cạnh file `.exe` cho phép người dùng thả trực tiếp model ONNX của mình vào mà không cần can thiệp mã nguồn.

## Kiểm thử

Chạy `.venv\Scripts\python.exe test_engine.py` và `.venv\Scripts\python.exe test_worker_ai.py`.