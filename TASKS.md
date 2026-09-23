# Kế hoạch phát triển phần mềm đổi giọng AI thời gian thực (Real-time Voice Changer)

---

## Giai đoạn 1: Core Audio Engine & I/O Pipeline
- [x] **Task 1.1**: Khởi tạo cấu trúc dự án (`core/`, `gui/`, `models/`, `voices/`, `utils/`).
- [x] **Task 1.2**: Xây dựng module thu/phát âm thanh qua `sounddevice` (WASAPI low-latency).
- [x] **Task 1.3**: Viết Circular RingBuffer để đệm luồng audio ổn định, không nấc tiếng.
- [x] **Task 1.4**: Cài đặt thuật toán Crossfade Overlap-Add gộp chunk mượt mà.
- [x] **Task 1.5**: Tích hợp module lọc nhiễu VAD (Voice Activity Detection) hoặc RNNoise.

---

## Giai đoạn 2: AI Pipeline & Đa giọng (Multi-Voice)
- [x] **Task 2.1**: Tích hợp module bóc tách cao độ (Pitch Extractor) `RMVPE` tối ưu ONNX và hỗ trợ semitone shifting (`models/rvc_pipeline.py`).
- [x] **Task 2.2**: Xây dựng inference pipeline mô hình RVC v2 với ONNX Runtime đa đầu vào: HuBERT feature + RMVPE F0 + Synthesizer (`models/rvc_pipeline.py`).
- [x] **Task 2.3**: Tạo module `VoiceManager`:
  - [x] Quét và nạp metadata từ thư mục `voices/` (`config.json`, avatar, model onnx, index faiss).
  - [x] Cơ chế Hot-swap đổi model giọng trong RAM/VRAM không ngắt luồng audio (< 50ms).
- [ ] **Task 2.4**: Tích hợp FAISS index search cải thiện chất âm đặc trưng giọng đích.
- [x] **Task 2.5**: Đo và tối ưu độ trễ tổng: tách audio callback và worker thread, đệm hàng đợi không nghẽn với drop frame khi nghẽn.

### Nâng cấp chất lượng & Kiến trúc AI
- [x] Chuẩn hóa audio 48 kHz, block 10 ms.
- [x] Thêm DC blocker, tone shaping stateful, compressor và soft limiter chống vỡ tiếng.
- [x] Thêm worker thread riêng biệt (`core/worker_pipeline.py`) bảo vệ PortAudio callback không bị giật lag khi tải nặng.
- [x] Xây dựng script nhập giọng RVC chuẩn hóa (`import_voice.py`) và endpoint API `/api/import_voice`, `/api/rescan_voices`.
- [x] Catalog giọng nam/nữ/anime rõ ràng với trạng thái model readiness và cảnh báo thiếu checkpoint thật.

---

## Giai đoạn 3: Giao diện người dùng hiện đại (Modern GUI)
- [x] **Task 3.1**: Khởi tạo khung ứng dụng GUI (Web Fluent Dark Mode, Local HTTP/REST Server).
- [x] **Task 3.2**: Thiết kế layout chuẩn Windows 11 Dark Mode:
  - [x] Header: Latency indicator, VAD active dot, Engine ON/OFF switch.
  - [x] Main Panel: Voice Cards Grid (Avatar, tên giọng, hotkey/pitch badge).
  - [x] Control Panel: Sliders (Pitch Shift -12 đến +12, Noise gate threshold, Monitor Volume).
  - [x] Footer: Audio device selectors (Mic in, VB-Cable out, Monitor out).
- [x] **Task 3.3**: Vẽ Canvas Audio Visualizer thời gian thực (Waveform động theo VAD).
- [x] **Task 3.4**: Xây dựng kênh giao tiếp IPC / REST API giữa GUI và Core Audio Engine.

---

## Giai đoạn 4: Audio Routing & Tích hợp hệ thống
- [x] **Task 4.1**: Cấu hình định tuyến ngõ ra vào Virtual Audio Cable (VB-CABLE) cho Discord/Game/OBS.
- [x] **Task 4.2**: Viết tính năng Direct Monitoring (nghe thử giọng qua tai nghe, nút chỉnh âm lượng riêng).
- [ ] **Task 4.3**: Bắt sự kiện Global Hotkeys chuyển giọng nhanh khi đang chơi game.
- [ ] **Task 4.4**: Bổ sung Soundboard mini (phát âm thanh định dạng `.wav`/`.mp3` gán phím tắt).

---

## Giai đoạn 5: Tối ưu, Kiểm thử & Đóng gói
- [ ] **Task 5.1**: Stress test chạy liên tục 4-8 tiếng kiểm tra rò rỉ bộ nhớ VRAM/RAM.
- [ ] **Task 5.2**: Xử lý fallback GPU: tự động chuyển cấu hình DirectML/CPU khi không có card NVIDIA.
- [ ] **Task 5.3**: Đóng gói Backend Python sang file thực thi độc lập (PyInstaller/Nuitka).
- [ ] **Task 5.4**: Tạo bộ cài đặt hoàn chỉnh Windows (`.exe` / `.msi`) tự kèm script cấu hình audio driver.
