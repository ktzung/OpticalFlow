import cv2
import numpy as np
import os

def enhance_optical_flow_visualization(flow_bgr, magnitude, enhance_contrast=True, 
                                       enhance_saturation=True, enhance_brightness=True):
    """
    Tăng cường chất lượng hiển thị optical flow
    
    Parameters:
    -----------
    flow_bgr : np.ndarray
        Ảnh optical flow ở định dạng BGR
    magnitude : np.ndarray
        Độ lớn của optical flow
    enhance_contrast : bool
        Tăng độ tương phản
    enhance_saturation : bool
        Tăng độ bão hòa màu
    enhance_brightness : bool
        Tăng độ sáng cho các vùng có chuyển động
    
    Returns:
    --------
    np.ndarray
        Ảnh optical flow đã được tăng cường
    """
    enhanced = flow_bgr.copy()
    
    # Chuyển sang HSV để dễ điều chỉnh
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Tăng độ bão hòa (Saturation) để màu sắc nổi bật hơn
    if enhance_saturation:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    
    # Tăng độ sáng (Value) dựa trên magnitude để làm nổi bật chuyển động
    if enhance_brightness:
        # Chuẩn hóa magnitude về 0-1
        mag_norm = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        # Tăng độ sáng cho các vùng có chuyển động mạnh
        brightness_boost = 1.0 + mag_norm * 0.5  # Tăng tối đa 50%
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_boost, 0, 255)
    
    # Chuyển lại BGR
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Tăng độ tương phản bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if enhance_contrast:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def create_enhanced_overlay(frame, flow_bgr, magnitude, flow, 
                           frame_weight=0.5, flow_weight=0.5, 
                           draw_arrows=True, arrow_step=20, arrow_thickness=2):
    """
    Tạo overlay chất lượng cao với optical flow nổi bật
    
    Parameters:
    -----------
    frame : np.ndarray
        Frame gốc
    flow_bgr : np.ndarray
        Ảnh optical flow
    magnitude : np.ndarray
        Độ lớn của flow
    flow : np.ndarray
        Flow vector
    frame_weight : float
        Trọng số của frame gốc (0-1)
    flow_weight : float
        Trọng số của optical flow (0-1)
    draw_arrows : bool
        Có vẽ mũi tên không
    arrow_step : int
        Khoảng cách giữa các mũi tên
    arrow_thickness : int
        Độ dày của mũi tên
    
    Returns:
    --------
    np.ndarray
        Overlay đã được tăng cường
    """
    # Tăng cường optical flow
    enhanced_flow = enhance_optical_flow_visualization(flow_bgr, magnitude)
    
    # Tạo overlay với tỷ lệ điều chỉnh (giảm frame gốc, tăng flow để nổi bật hơn)
    overlay = cv2.addWeighted(frame, frame_weight, enhanced_flow, flow_weight, 0)
    
    # Vẽ mũi tên để hiển thị hướng flow
    if draw_arrows:
        # Tính threshold động dựa trên magnitude trung bình
        mag_mean = np.mean(magnitude)
        mag_threshold = max(mag_mean * 0.5, 2.0)  # Threshold tối thiểu là 2.0
        
        for y in range(0, frame.shape[0], arrow_step):
            for x in range(0, frame.shape[1], arrow_step):
                fx, fy = flow[y, x]
                mag = magnitude[y, x]
                
                # Chỉ vẽ mũi tên nếu có chuyển động đáng kể
                if mag > mag_threshold:
                    # Tính độ dài mũi tên (giới hạn để không quá dài)
                    arrow_length = min(mag * 2, 30)
                    if arrow_length > 3:
                        # Tính điểm kết thúc
                        end_x = int(x + fx * arrow_length / mag)
                        end_y = int(y + fy * arrow_length / mag)
                        
                        # Màu mũi tên dựa trên hướng (lấy từ flow tại điểm đó)
                        arrow_color = tuple(map(int, enhanced_flow[y, x]))
                        
                        # Vẽ mũi tên với độ dày và độ dài phù hợp
                        cv2.arrowedLine(
                            overlay,
                            (x, y),
                            (end_x, end_y),
                            arrow_color,
                            arrow_thickness,
                            tipLength=0.4,
                            line_type=cv2.LINE_AA  # Anti-aliasing cho đường mượt hơn
                        )
    
    # Tăng độ sắc nét (sharpening) cho overlay
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(overlay, -1, kernel * 0.3)  # Nhẹ nhàng để không quá mạnh
    overlay = cv2.addWeighted(overlay, 0.7, sharpened, 0.3, 0)
    
    return overlay


def calculate_optical_flow_farneback(video_path, output_dir='output_frames', visualize=True):
    """
    Tính toán và minh họa luồng quang học (optical flow) sử dụng phương pháp Farneback
    
    Parameters:
    -----------
    video_path : str
        Đường dẫn đến file video đầu vào
    output_dir : str
        Thư mục để lưu các frame đã xử lý
    visualize : bool
        Có hiển thị kết quả hay không
    """
    
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return
    
    # Đọc frame đầu tiên và chuyển sang grayscale
    ret, frame1 = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame từ video")
        cap.release()
        return
    
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Tạo mask HSV để vẽ optical flow (màu sắc)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # Độ bão hòa = 255
    
    frame_count = 0
    
    print("Đang xử lý video...")
    
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        # Chuyển frame hiện tại sang grayscale
        curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Tính toán optical flow sử dụng phương pháp Farneback
        # Parameters:
        # - prev_gray: frame trước đó (grayscale)
        # - curr_gray: frame hiện tại (grayscale)
        # - None: pyramid scale (0.5 = mỗi level giảm 50%)
        # - levels: số lượng pyramid levels
        # - winsize: kích thước cửa sổ trung bình
        # - iterations: số lần lặp ở mỗi pyramid level
        # - poly_n: kích thước neighborhood để tìm đa thức
        # - poly_sigma: độ lệch chuẩn Gaussian để làm mịn
        # - flags: cờ điều khiển
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, 
            curr_gray, 
            None,
            0.5,      # pyr_scale: tỷ lệ pyramid
            3,        # levels: số lượng pyramid levels
            15,       # winsize: kích thước cửa sổ trung bình
            3,        # iterations: số lần lặp
            5,        # poly_n: kích thước neighborhood
            1.2,      # poly_sigma: độ lệch chuẩn Gaussian
            0         # flags
        )
        
        # Chuyển đổi flow thành tọa độ cực (magnitude và angle)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Tạo hình ảnh màu để hiển thị optical flow
        # Hue (màu sắc) dựa trên hướng của flow
        hsv[..., 0] = angle * 180 / np.pi / 2
        
        # Value (độ sáng) dựa trên độ lớn của flow (chuẩn hóa với gamma correction)
        # Sử dụng gamma correction để làm nổi bật các chuyển động nhỏ hơn
        mag_normalized = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        mag_gamma = np.power(mag_normalized, 0.7)  # Gamma < 1 để tăng độ sáng
        hsv[..., 2] = (mag_gamma * 255).astype(np.uint8)
        
        # Chuyển HSV sang BGR để hiển thị
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Kết hợp frame gốc với optical flow (tùy chọn)
        if visualize:
            # Tạo overlay chất lượng cao với optical flow nổi bật
            overlay = create_enhanced_overlay(
                frame2, 
                flow_bgr, 
                magnitude, 
                flow,
                frame_weight=0.5,      # Giảm frame gốc xuống 50% để flow nổi bật hơn
                flow_weight=0.5,       # Tăng flow lên 50%
                draw_arrows=True,
                arrow_step=20,         # Khoảng cách mũi tên
                arrow_thickness=2      # Độ dày mũi tên
            )
            
            # Tăng cường optical flow thuần túy
            enhanced_flow = enhance_optical_flow_visualization(flow_bgr, magnitude)
            
            # Lưu frame gốc (chưa có optical flow) với chất lượng cao
            original_path = os.path.join(output_dir, f'frame_{frame_count:04d}_original.jpg')
            cv2.imwrite(original_path, frame2, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Lưu frame overlay với chất lượng cao (JPEG quality 95)
            output_path = os.path.join(output_dir, f'frame_{frame_count:04d}_overlay.jpg')
            cv2.imwrite(output_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Lưu optical flow thuần túy với chất lượng cao
            flow_path = os.path.join(output_dir, f'frame_{frame_count:04d}_flow.jpg')
            cv2.imwrite(flow_path, enhanced_flow, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Cập nhật frame trước đó
        prev_gray = curr_gray.copy()
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} frames...")
    
    cap.release()
    print(f"\nHoàn thành! Đã xử lý {frame_count} frames.")
    print(f"Kết quả được lưu trong thư mục: {output_dir}")
    
    return frame_count


def visualize_optical_flow_realtime(video_path):
    """
    Hiển thị optical flow theo thời gian thực (real-time)
    
    Parameters:
    -----------
    video_path : str
        Đường dẫn đến file video đầu vào
    """
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return
    
    ret, frame1 = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame từ video")
        cap.release()
        return
    
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    
    print("Nhấn 'q' để thoát, 'Space' để tạm dừng")
    
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Tính toán optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Chuyển đổi sang HSV
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        mag_normalized = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        mag_gamma = np.power(mag_normalized, 0.7)
        hsv[..., 2] = (mag_gamma * 255).astype(np.uint8)
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Tăng cường optical flow
        enhanced_flow = enhance_optical_flow_visualization(flow_bgr, magnitude)
        
        # Hiển thị
        cv2.imshow('Original Frame', frame2)
        cv2.imshow('Optical Flow (Farneback)', enhanced_flow)
        
        # Kết hợp và hiển thị với chất lượng cao
        overlay = create_enhanced_overlay(
            frame2, enhanced_flow, magnitude, flow,
            frame_weight=0.5, flow_weight=0.5, draw_arrows=True
        )
        cv2.imshow('Overlay', overlay)
        
        prev_gray = curr_gray.copy()
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)  # Tạm dừng
    
    cap.release()
    cv2.destroyAllWindows()


def create_thesis_illustration(output_dir='output_frames', num_frames=16, 
                                rows=2, cols=8, frame_type='overlay', 
                                output_name='thesis_illustration.jpg'):
    """
    Chọn các frames đại diện và ghép thành 1 ảnh để minh họa trong thesis
    
    Parameters:
    -----------
    output_dir : str
        Thư mục chứa các frames đã xử lý
    num_frames : int
        Số lượng frames cần chọn (mặc định 16)
    rows : int
        Số hàng trong ảnh kết quả (mặc định 2)
    cols : int
        Số cột trong ảnh kết quả (mặc định 8)
    frame_type : str
        Loại frame: 'overlay', 'flow', hoặc 'original' (mặc định 'overlay')
        - 'overlay': frame gốc + optical flow
        - 'flow': optical flow thuần túy
        - 'original': frame gốc (chưa có optical flow)
    output_name : str
        Tên file ảnh kết quả
    """
    
    if not os.path.exists(output_dir):
        print(f"Lỗi: Thư mục {output_dir} không tồn tại!")
        print("Vui lòng chạy xử lý video trước (chế độ 1).")
        return
    
    # Tìm tất cả các file frame
    frame_files = []
    if frame_type == 'overlay':
        pattern = '_overlay.jpg'
    elif frame_type == 'flow':
        pattern = '_flow.jpg'
    elif frame_type == 'original':
        pattern = '_original.jpg'
    else:
        print(f"Lỗi: Loại frame không hợp lệ: {frame_type}")
        return
    
    for filename in os.listdir(output_dir):
        if filename.endswith(pattern) and filename.startswith('frame_'):
            frame_files.append(filename)
    
    if len(frame_files) == 0:
        print(f"Lỗi: Không tìm thấy frames trong thư mục {output_dir}!")
        return
    
    # Sắp xếp theo tên file (theo số thứ tự frame)
    frame_files.sort()
    
    total_frames = len(frame_files)
    print(f"Tìm thấy {total_frames} frames trong thư mục {output_dir}")
    
    # Chọn các frames đại diện (chia đều từ đầu đến cuối)
    if total_frames < num_frames:
        print(f"Cảnh báo: Chỉ có {total_frames} frames, sẽ sử dụng tất cả.")
        selected_frames = frame_files
    else:
        # Chọn frames cách đều nhau
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        selected_frames = [frame_files[i] for i in indices]
    
    print(f"Đã chọn {len(selected_frames)} frames đại diện")
    
    # Đọc frame đầu tiên để lấy kích thước
    first_frame_path = os.path.join(output_dir, selected_frames[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"Lỗi: Không thể đọc frame {selected_frames[0]}")
        return
    
    h, w = first_frame.shape[:2]
    print(f"Kích thước mỗi frame: {w}x{h}")
    
    # Tạo ảnh kết quả: rows hàng x cols cột
    result_height = h * rows
    result_width = w * cols
    result_image = np.zeros((result_height, result_width, 3), dtype=np.uint8)
    
    # Điền các frames vào ảnh kết quả
    for idx, frame_file in enumerate(selected_frames):
        row = idx // cols
        col = idx % cols
        
        frame_path = os.path.join(output_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Cảnh báo: Không thể đọc {frame_file}, bỏ qua...")
            continue
        
        # Resize frame nếu cần (đảm bảo cùng kích thước)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        
        # Tính toán vị trí trong ảnh kết quả
        y_start = row * h
        y_end = y_start + h
        x_start = col * w
        x_end = x_start + w
        
        # Đặt frame vào vị trí tương ứng
        result_image[y_start:y_end, x_start:x_end] = frame
        
        # Vẽ số thứ tự frame (tùy chọn)
        frame_num = frame_file.split('_')[1].split('_')[0]
        cv2.putText(result_image, f'Frame {frame_num}', 
                   (x_start + 10, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_image, f'Frame {frame_num}', 
                   (x_start + 10, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Lưu ảnh kết quả với chất lượng cao
    output_path = os.path.join(output_dir, output_name)
    cv2.imwrite(output_path, result_image, [cv2.IMWRITE_JPEG_QUALITY, 98])
    
    print(f"\n✓ Đã tạo ảnh minh họa: {output_path}")
    print(f"  Kích thước: {result_width}x{result_height} pixels")
    print(f"  Layout: {rows} hàng x {cols} cột")
    print(f"  Loại frame: {frame_type}")
    
    return output_path


if __name__ == "__main__":
    import sys
    import argparse
    
    # Thiết lập parser cho tham số dòng lệnh
    parser = argparse.ArgumentParser(
        description='Tính toán và minh họa Optical Flow sử dụng phương pháp Farneback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python optical_flow_farneback.py 04.mp4
  python optical_flow_farneback.py 04.mp4 --mode 1 --output output_frames_04
  python optical_flow_farneback.py 04.mp4 --mode 1 --auto
  python optical_flow_farneback.py 04.mp4 --mode 3 --type overlay
        """
    )
    
    parser.add_argument(
        'video',
        nargs='?',
        help='Đường dẫn đến file video (ví dụ: 04.mp4). Nếu không chỉ định, sẽ liệt kê các video có sẵn.'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=int,
        choices=[1, 2, 3],
        help='Chế độ xử lý: 1=xử lý và lưu frames, 2=hiển thị real-time, 3=tạo ảnh minh họa'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Thư mục output (mặc định: output_frames hoặc output_frames_<số_video>)'
    )
    
    parser.add_argument(
        '--auto', '-a',
        action='store_true',
        help='Tự động chạy không cần hỏi (chỉ dùng với --mode 1)'
    )
    
    parser.add_argument(
        '--type', '-t',
        choices=['overlay', 'flow', 'original'],
        default='overlay',
        help='Loại frame khi dùng --mode 3 (mặc định: overlay)'
    )
    
    parser.add_argument(
        '--frames', '-f',
        type=int,
        default=16,
        help='Số lượng frames cho ảnh minh họa (mặc định: 16)'
    )
    
    args = parser.parse_args()
    
    # Xác định video path
    if args.video:
        video_path = args.video
    else:
        # Tìm video có sẵn trong thư mục
        video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
        
        if not video_files:
            print("Lỗi: Không tìm thấy file video nào trong thư mục hiện tại.")
            print("Vui lòng đảm bảo có file video (.mp4) trong thư mục.")
            print("\nSử dụng: python optical_flow_farneback.py <video_file>")
            sys.exit(1)
        
        # Nếu có nhiều video, hỏi người dùng chọn
        if len(video_files) > 1:
            print("\nTìm thấy các video sau:")
            for i, vf in enumerate(video_files, 1):
                print(f"  {i}. {vf}")
            
            try:
                choice = input(f"\nChọn video (1-{len(video_files)}) hoặc nhập tên file: ").strip()
                # Kiểm tra nếu là số
                if choice.isdigit() and 1 <= int(choice) <= len(video_files):
                    video_path = video_files[int(choice) - 1]
                else:
                    # Nếu không phải số, coi như tên file
                    video_path = choice if choice.endswith('.mp4') else choice + '.mp4'
            except (ValueError, KeyboardInterrupt):
                print("\nSử dụng video mặc định: 03.mp4")
                video_path = "03.mp4"
        else:
            video_path = video_files[0]
            print(f"Sử dụng video: {video_path}")
    
    # Kiểm tra file video có tồn tại không
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video {video_path}")
        print("Vui lòng đảm bảo file video tồn tại trong thư mục hiện tại.")
        sys.exit(1)
    
    # Xác định thư mục output
    if args.output:
        output_dir = args.output
    else:
        # Tự động tạo tên thư mục dựa trên tên video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f'output_frames_{video_name}' if video_name.isdigit() else 'output_frames'
    
    # Xử lý theo mode
    if args.mode:
        # Chế độ được chỉ định qua tham số
        if args.mode == 1:
            # Xử lý và lưu frames
            total_frames = calculate_optical_flow_farneback(video_path, output_dir=output_dir, visualize=True)
            
            if args.auto:
                # Tự động tạo tất cả ảnh minh họa
                if total_frames and total_frames > 0:
                    print("\nĐang tạo ảnh minh họa tự động...")
                    create_thesis_illustration(
                        output_dir=output_dir,
                        num_frames=args.frames,
                        rows=2,
                        cols=8,
                        frame_type='overlay',
                        output_name='thesis_illustration.jpg'
                    )
                    create_thesis_illustration(
                        output_dir=output_dir,
                        num_frames=args.frames,
                        rows=2,
                        cols=8,
                        frame_type='flow',
                        output_name='thesis_illustration_flow.jpg'
                    )
                    create_thesis_illustration(
                        output_dir=output_dir,
                        num_frames=args.frames,
                        rows=2,
                        cols=8,
                        frame_type='original',
                        output_name='thesis_illustration_original.jpg'
                    )
            elif total_frames and total_frames > 0:
                # Hỏi có muốn tạo ảnh minh họa không
                create_illustration = input("\nBạn có muốn tạo ảnh minh họa cho thesis? (y/n): ").strip().lower()
                if create_illustration == 'y':
                    print("\nĐang tạo ảnh minh họa...")
                    create_thesis_illustration(
                        output_dir=output_dir,
                        num_frames=args.frames,
                        rows=2,
                        cols=8,
                        frame_type='overlay',
                        output_name='thesis_illustration.jpg'
                    )
                    
                    create_flow = input("\nBạn có muốn tạo ảnh minh họa với optical flow thuần túy? (y/n): ").strip().lower()
                    if create_flow == 'y':
                        create_thesis_illustration(
                            output_dir=output_dir,
                            num_frames=args.frames,
                            rows=2,
                            cols=8,
                            frame_type='flow',
                            output_name='thesis_illustration_flow.jpg'
                        )
                    
                    create_original = input("\nBạn có muốn tạo ảnh minh họa từ frame gốc? (y/n): ").strip().lower()
                    if create_original == 'y':
                        create_thesis_illustration(
                            output_dir=output_dir,
                            num_frames=args.frames,
                            rows=2,
                            cols=8,
                            frame_type='original',
                            output_name='thesis_illustration_original.jpg'
                        )
        
        elif args.mode == 2:
            # Hiển thị real-time
            visualize_optical_flow_realtime(video_path)
        
        elif args.mode == 3:
            # Tạo ảnh minh họa từ frames đã có
            output_name = f'thesis_illustration_{args.type}.jpg'
            create_thesis_illustration(
                output_dir=output_dir,
                num_frames=args.frames,
                rows=2,
                cols=8,
                frame_type=args.type,
                output_name=output_name
            )
    else:
        # Chế độ tương tác (như cũ)
        print("=" * 60)
        print("THUẬT TOÁN OPTICAL FLOW SỬ DỤNG PHƯƠNG PHÁP FARNEBACK")
        print("=" * 60)
        print("\nChọn chế độ:")
        print("1. Xử lý và lưu tất cả frames vào thư mục output")
        print("2. Hiển thị real-time (không lưu)")
        print("3. Tạo ảnh minh họa cho thesis (16 frames, 2 hàng x 8 cột)")
        
        choice = input("\nNhập lựa chọn (1, 2 hoặc 3): ").strip()
        
        if choice == "1":
            # Xử lý và lưu tất cả frames
            total_frames = calculate_optical_flow_farneback(video_path, output_dir=output_dir, visualize=True)
            
            # Hỏi có muốn tạo ảnh minh họa không
            if total_frames and total_frames > 0:
                create_illustration = input("\nBạn có muốn tạo ảnh minh họa cho thesis? (y/n): ").strip().lower()
                if create_illustration == 'y':
                    print("\nĐang tạo ảnh minh họa...")
                    create_thesis_illustration(
                        output_dir=output_dir,
                        num_frames=16,
                        rows=2,
                        cols=8,
                        frame_type='overlay',
                        output_name='thesis_illustration.jpg'
                    )
                    
                    # Hỏi có muốn tạo thêm ảnh flow thuần túy không
                    create_flow = input("\nBạn có muốn tạo ảnh minh họa với optical flow thuần túy? (y/n): ").strip().lower()
                    if create_flow == 'y':
                        create_thesis_illustration(
                            output_dir=output_dir,
                            num_frames=16,
                            rows=2,
                            cols=8,
                            frame_type='flow',
                            output_name='thesis_illustration_flow.jpg'
                        )
                    
                    # Hỏi có muốn tạo ảnh từ frame gốc (chưa có optical flow) không
                    create_original = input("\nBạn có muốn tạo ảnh minh họa từ frame gốc (chưa có optical flow)? (y/n): ").strip().lower()
                    if create_original == 'y':
                        create_thesis_illustration(
                            output_dir=output_dir,
                            num_frames=16,
                            rows=2,
                            cols=8,
                            frame_type='original',
                            output_name='thesis_illustration_original.jpg'
                        )
        
        elif choice == "2":
            # Hiển thị real-time
            visualize_optical_flow_realtime(video_path)
        
        elif choice == "3":
            # Tạo ảnh minh họa từ frames đã có
            print("\nChọn loại frame:")
            print("1. Overlay (frame gốc + optical flow)")
            print("2. Flow (optical flow thuần túy)")
            print("3. Original (frame gốc, chưa có optical flow)")
            frame_choice = input("Nhập lựa chọn (1, 2 hoặc 3): ").strip()
            
            if frame_choice == '1':
                frame_type = 'overlay'
                output_name = 'thesis_illustration.jpg'
            elif frame_choice == '2':
                frame_type = 'flow'
                output_name = 'thesis_illustration_flow.jpg'
            elif frame_choice == '3':
                frame_type = 'original'
                output_name = 'thesis_illustration_original.jpg'
            else:
                print("Lựa chọn không hợp lệ. Sử dụng overlay mặc định.")
                frame_type = 'overlay'
                output_name = 'thesis_illustration.jpg'
            
            create_thesis_illustration(
                output_dir=output_dir,
                num_frames=16,
                rows=2,
                cols=8,
                frame_type=frame_type,
                output_name=output_name
            )
        
        else:
            print("Lựa chọn không hợp lệ. Chạy chế độ mặc định (lưu frames)...")
            calculate_optical_flow_farneback(video_path, output_dir='output_frames', visualize=True)
