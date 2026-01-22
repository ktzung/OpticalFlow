"""
Script để xử lý video 45678 (04, 05, 06, 07, 08) với optical flow
"""
import os
from optical_flow_farneback import calculate_optical_flow_farneback, create_thesis_illustration

def process_video(video_name, video_number):
    """
    Xử lý một video và tạo các ảnh minh họa
    
    Parameters:
    -----------
    video_name : str
        Tên file video (ví dụ: "04.mp4")
    video_number : str
        Số thứ tự video (ví dụ: "04")
    """
    print("\n" + "=" * 70)
    print(f"XỬ LÝ VIDEO: {video_name}")
    print("=" * 70)
    
    if not os.path.exists(video_name):
        print(f"⚠ Cảnh báo: Không tìm thấy file {video_name}, bỏ qua...")
        return False
    
    # Tạo thư mục output riêng cho mỗi video
    output_dir = f'output_frames_{video_number}'
    
    # Xử lý video
    total_frames = calculate_optical_flow_farneback(
        video_name, 
        output_dir=output_dir, 
        visualize=True
    )
    
    if total_frames and total_frames > 0:
        print(f"\n{'='*70}")
        print(f"TẠO ẢNH MINH HỌA CHO VIDEO {video_number}")
        print(f"{'='*70}")
        
        # Tạo ảnh minh họa overlay
        print(f"\n[{video_number}] Đang tạo ảnh minh họa với overlay...")
        create_thesis_illustration(
            output_dir=output_dir,
            num_frames=16,
            rows=2,
            cols=8,
            frame_type='overlay',
            output_name=f'thesis_illustration_{video_number}.jpg'
        )
        
        # Tạo ảnh minh họa flow thuần túy
        print(f"\n[{video_number}] Đang tạo ảnh minh họa với optical flow thuần túy...")
        create_thesis_illustration(
            output_dir=output_dir,
            num_frames=16,
            rows=2,
            cols=8,
            frame_type='flow',
            output_name=f'thesis_illustration_flow_{video_number}.jpg'
        )
        
        # Tạo ảnh minh họa từ frame gốc
        print(f"\n[{video_number}] Đang tạo ảnh minh họa từ frame gốc...")
        create_thesis_illustration(
            output_dir=output_dir,
            num_frames=16,
            rows=2,
            cols=8,
            frame_type='original',
            output_name=f'thesis_illustration_original_{video_number}.jpg'
        )
        
        print(f"\n✓ Hoàn thành video {video_number}!")
        print(f"  - Đã xử lý {total_frames} frames")
        print(f"  - Kết quả trong: {output_dir}/")
        return True
    else:
        print(f"⚠ Lỗi khi xử lý video {video_number}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("XỬ LÝ NHIỀU VIDEO VỚI OPTICAL FLOW - PHƯƠNG PHÁP FARNEBACK")
    print("=" * 70)
    print("\nDanh sách video sẽ xử lý: 04, 05, 06, 07, 08")
    print("(Bỏ qua video 03 vì đã xử lý trước đó)")
    
    # Danh sách video cần xử lý
    videos_to_process = [
        ("04.mp4", "04"),
        ("05.mp4", "05"),
        ("06.mp4", "06"),
        ("07.mp4", "07"),
        ("08.mp4", "08")
    ]
    
    # Xử lý từng video
    success_count = 0
    total_videos = len(videos_to_process)
    
    for video_name, video_number in videos_to_process:
        if process_video(video_name, video_number):
            success_count += 1
    
    # Tóm tắt kết quả
    print("\n" + "=" * 70)
    print("TÓM TẮT KẾT QUẢ")
    print("=" * 70)
    print(f"\nĐã xử lý thành công: {success_count}/{total_videos} video")
    
    if success_count > 0:
        print("\nCác thư mục output:")
        for _, video_number in videos_to_process:
            output_dir = f'output_frames_{video_number}'
            if os.path.exists(output_dir):
                print(f"  ✓ {output_dir}/")
                print(f"    - thesis_illustration_{video_number}.jpg")
                print(f"    - thesis_illustration_flow_{video_number}.jpg")
                print(f"    - thesis_illustration_original_{video_number}.jpg")
    
    print("\n" + "=" * 70)
    print("HOÀN THÀNH TẤT CẢ!")
    print("=" * 70)
