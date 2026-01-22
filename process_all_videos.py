"""
Script để xử lý tất cả 12 video (01-12) với optical flow
Mỗi video sẽ có thư mục output riêng và 3 phiên bản ảnh minh họa
"""
import os
import sys
from optical_flow_farneback import calculate_optical_flow_farneback, create_thesis_illustration

def process_video(video_name, video_number):
    """
    Xử lý một video và tạo các ảnh minh họa
    
    Parameters:
    -----------
    video_name : str
        Tên file video (ví dụ: "01.mp4")
    video_number : str
        Số thứ tự video (ví dụ: "01")
    
    Returns:
    --------
    bool
        True nếu thành công, False nếu thất bại
    """
    print("\n" + "=" * 80)
    print(f"XỬ LÝ VIDEO: {video_name} ({video_number}/12)")
    print("=" * 80)
    
    if not os.path.exists(video_name):
        print(f"⚠ Cảnh báo: Không tìm thấy file {video_name}, bỏ qua...")
        return False
    
    # Tạo thư mục output riêng cho mỗi video
    output_dir = f'output_frames_{video_number}'
    
    try:
        # Xử lý video
        print(f"\n[{video_number}] Đang xử lý video {video_name}...")
        total_frames = calculate_optical_flow_farneback(
            video_name, 
            output_dir=output_dir, 
            visualize=True
        )
        
        if total_frames and total_frames > 0:
            print(f"\n{'='*80}")
            print(f"TẠO ẢNH MINH HỌA CHO VIDEO {video_number}")
            print(f"{'='*80}")
            
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
            print(f"  - Thư mục output: {output_dir}/")
            print(f"  - 3 ảnh minh họa đã được tạo")
            return True
        else:
            print(f"⚠ Lỗi: Không thể xử lý video {video_number}")
            return False
            
    except Exception as e:
        print(f"⚠ Lỗi khi xử lý video {video_number}: {str(e)}")
        return False


def main():
    """Hàm chính để xử lý tất cả video"""
    print("=" * 80)
    print("XỬ LÝ TẤT CẢ 12 VIDEO VỚI OPTICAL FLOW - PHƯƠNG PHÁP FARNEBACK")
    print("=" * 80)
    print("\nDanh sách video sẽ xử lý: 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12")
    print("Mỗi video sẽ có:")
    print("  - Thư mục output riêng (output_frames_XX)")
    print("  - 3 ảnh minh họa: overlay, flow, original")
    print("\nBắt đầu xử lý...\n")
    
    # Danh sách tất cả video cần xử lý
    videos_to_process = [
        ("01.mp4", "01"),
        ("02.mp4", "02"),
        ("03.mp4", "03"),
        ("04.mp4", "04"),
        ("05.mp4", "05"),
        ("06.mp4", "06"),
        ("07.mp4", "07"),
        ("08.mp4", "08"),
        ("09.mp4", "09"),
        ("10.mp4", "10"),
        ("11.mp4", "11"),
        ("12.mp4", "12")
    ]
    
    # Xử lý từng video
    success_count = 0
    failed_videos = []
    total_videos = len(videos_to_process)
    
    for idx, (video_name, video_number) in enumerate(videos_to_process, 1):
        print(f"\n{'='*80}")
        print(f"TIẾN ĐỘ: {idx}/{total_videos} video")
        print(f"{'='*80}")
        
        if process_video(video_name, video_number):
            success_count += 1
        else:
            failed_videos.append(video_number)
    
    # Tóm tắt kết quả
    print("\n" + "=" * 80)
    print("TÓM TẮT KẾT QUẢ")
    print("=" * 80)
    print(f"\nĐã xử lý thành công: {success_count}/{total_videos} video")
    
    if failed_videos:
        print(f"\n⚠ Các video thất bại: {', '.join(failed_videos)}")
    
    if success_count > 0:
        print("\n✓ Các thư mục output đã được tạo:")
        for _, video_number in videos_to_process:
            output_dir = f'output_frames_{video_number}'
            if os.path.exists(output_dir):
                print(f"  ✓ {output_dir}/")
                print(f"    - thesis_illustration_{video_number}.jpg (overlay)")
                print(f"    - thesis_illustration_flow_{video_number}.jpg (flow)")
                print(f"    - thesis_illustration_original_{video_number}.jpg (original)")
                print()
    
    print("=" * 80)
    if success_count == total_videos:
        print("✓ HOÀN THÀNH TẤT CẢ VIDEO!")
    else:
        print(f"⚠ HOÀN THÀNH VỚI {success_count}/{total_videos} VIDEO THÀNH CÔNG")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Đã dừng xử lý do người dùng hủy (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n⚠ Lỗi không mong muốn: {str(e)}")
        sys.exit(1)
