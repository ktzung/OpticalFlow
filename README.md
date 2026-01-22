# Thuật toán Optical Flow sử dụng phương pháp Farneback

Dự án này cung cấp thuật toán tính toán và minh họa luồng quang học (optical flow) cho các khung hình từ video đầu vào sử dụng phương pháp Farneback của OpenCV.

## Yêu cầu

- Python 3.6 trở lên
- OpenCV (cv2)
- NumPy

## Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Xử lý một video

#### Chạy script chính (chế độ tương tác)

```bash
python optical_flow_farneback.py
```

Hoặc chỉ định video cụ thể:

```bash
python optical_flow_farneback.py 04.mp4
```

Script sẽ hỏi bạn chọn chế độ:
1. **Xử lý và lưu tất cả frames**: Xử lý toàn bộ video và lưu các frame đã xử lý vào thư mục `output_frames`
2. **Hiển thị real-time**: Hiển thị kết quả optical flow theo thời gian thực (không lưu file)
3. **Tạo ảnh minh họa**: Tạo ảnh minh họa từ frames đã xử lý trước đó

#### Sử dụng tham số dòng lệnh (không tương tác)

**Xem trợ giúp:**
```bash
python optical_flow_farneback.py --help
```

**Các ví dụ sử dụng:**

```bash
# Xử lý video và tự động tạo tất cả ảnh minh họa
python optical_flow_farneback.py 04.mp4 --mode 1 --auto

# Xử lý video với thư mục output tùy chỉnh
python optical_flow_farneback.py 04.mp4 --mode 1 --output output_frames_04

# Chỉ tạo ảnh minh họa từ frames đã có
python optical_flow_farneback.py 04.mp4 --mode 3 --type overlay --output output_frames_04

# Tạo ảnh minh họa với số frames tùy chỉnh
python optical_flow_farneback.py 04.mp4 --mode 3 --type flow --frames 20 --output output_frames_04

# Hiển thị real-time
python optical_flow_farneback.py 04.mp4 --mode 2
```

**Các tham số dòng lệnh:**

- `video`: Đường dẫn đến file video (vị trí đầu tiên)
- `--mode`, `-m`: Chế độ xử lý (1=xử lý và lưu, 2=real-time, 3=tạo ảnh minh họa)
- `--output`, `-o`: Thư mục output (mặc định: `output_frames_<số_video>`)
- `--auto`, `-a`: Tự động chạy không cần hỏi (chỉ dùng với `--mode 1`)
- `--type`, `-t`: Loại frame khi dùng `--mode 3` (overlay/flow/original)
- `--frames`, `-f`: Số lượng frames cho ảnh minh họa (mặc định: 16)

#### Chạy script test tự động

```bash
python test_optical_flow.py
```

Script này sẽ tự động xử lý video và tạo tất cả ảnh minh họa.

### 2. Xử lý nhiều video cùng lúc

#### Xử lý tất cả 12 video (01-12)

Để xử lý tất cả 12 video cùng lúc:

**Windows:**
```bash
run_all_videos.bat
```

**Linux/Mac hoặc Python trực tiếp:**
```bash
python process_all_videos.py
```

Script này sẽ:
- Tự động xử lý tất cả video từ 01.mp4 đến 12.mp4
- Tạo thư mục output riêng cho mỗi video: `output_frames_01/`, `output_frames_02/`, ..., `output_frames_12/`
- Tạo 3 ảnh minh họa cho mỗi video:
  - `thesis_illustration_XX.jpg` (overlay)
  - `thesis_illustration_flow_XX.jpg` (flow)
  - `thesis_illustration_original_XX.jpg` (original)

#### Xử lý một nhóm video cụ thể

Để xử lý một nhóm video (ví dụ: 04, 05, 06, 07, 08):

```bash
python process_video_45678.py
```

Script này sẽ:
- Tự động xử lý các video 04.mp4, 05.mp4, 06.mp4, 07.mp4, 08.mp4
- Tạo thư mục output riêng cho mỗi video: `output_frames_04/`, `output_frames_05/`, ...
- Tạo 3 ảnh minh họa cho mỗi video (overlay, flow, original)

### Cấu trúc thư mục output

Khi chọn chế độ 1, các file sau sẽ được tạo trong thư mục `output_frames`:
- `frame_XXXX_original.jpg`: Frame gốc (chưa có optical flow) - để so sánh
- `frame_XXXX_overlay.jpg`: Frame gốc kết hợp với optical flow (có mũi tên chỉ hướng)
- `frame_XXXX_flow.jpg`: Optical flow thuần túy (màu sắc biểu thị hướng và độ lớn)

#### Ảnh minh họa cho thesis

Khi tạo ảnh minh họa, bạn sẽ có:
- `thesis_illustration.jpg`: 16 frames overlay (2 hàng × 8 cột)
- `thesis_illustration_flow.jpg`: 16 frames flow thuần túy
- `thesis_illustration_original.jpg`: 16 frames gốc (để so sánh)

## Giải thích phương pháp Farneback

Phương pháp Farneback là một thuật toán tính toán optical flow dựa trên:
- **Pyramid approach**: Sử dụng image pyramid để xử lý ở nhiều độ phân giải
- **Polynomial expansion**: Mô hình hóa chuyển động bằng đa thức
- **Dense flow**: Tính toán flow vector cho mọi pixel trong ảnh

### Tham số chính:

- `pyr_scale` (0.5): Tỷ lệ giảm kích thước ở mỗi level của pyramid
- `levels` (3): Số lượng levels trong pyramid
- `winsize` (15): Kích thước cửa sổ trung bình
- `iterations` (3): Số lần lặp ở mỗi pyramid level
- `poly_n` (5): Kích thước neighborhood để tìm đa thức
- `poly_sigma` (1.2): Độ lệch chuẩn Gaussian để làm mịn

## Cách đọc kết quả

- **Màu sắc**: Biểu thị hướng của chuyển động
  - Đỏ: Chuyển động sang phải
  - Xanh lá: Chuyển động lên trên
  - Xanh dương: Chuyển động xuống dưới
  - Vàng: Chuyển động sang trái
- **Độ sáng**: Biểu thị độ lớn của chuyển động (càng sáng = chuyển động càng nhanh)
- **Mũi tên**: Chỉ hướng và độ lớn của chuyển động tại các điểm mẫu

## Tùy chỉnh

Bạn có thể điều chỉnh các tham số trong hàm `cv2.calcOpticalFlowFarneback()` để phù hợp với video của mình:
- Tăng `winsize` để làm mịn hơn nhưng chậm hơn
- Tăng `levels` để xử lý tốt hơn với chuyển động lớn
- Điều chỉnh `poly_n` và `poly_sigma` để thay đổi độ nhạy
