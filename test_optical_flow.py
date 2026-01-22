"""
Script test để chạy thử optical flow tự động
"""
import os
from optical_flow_farneback import calculate_optical_flow_farneback, create_thesis_illustration

if __name__ == "__main__":
    video_path = "03.mp4"
    
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video {video_path}")
    else:
        print("=" * 60)
        print("CHẠY THỬ OPTICAL FLOW - PHƯƠNG PHÁP FARNEBACK")
        print("=" * 60)
        print(f"\nĐang xử lý video: {video_path}")
        print("Chế độ: Xử lý và lưu frames với chất lượng cao\n")
        
        # Xử lý video
        total_frames = calculate_optical_flow_farneback(
            video_path, 
            output_dir='output_frames', 
            visualize=True
        )
        
        if total_frames and total_frames > 0:
            print("\n" + "=" * 60)
            print("TẠO ẢNH MINH HỌA CHO THESIS")
            print("=" * 60)
            
            # Tạo ảnh minh họa overlay
            print("\nĐang tạo ảnh minh họa với overlay...")
            create_thesis_illustration(
                output_dir='output_frames',
                num_frames=16,
                rows=2,
                cols=8,
                frame_type='overlay',
                output_name='thesis_illustration.jpg'
            )
            
            # Tạo ảnh minh họa flow thuần túy
            print("\nĐang tạo ảnh minh họa với optical flow thuần túy...")
            create_thesis_illustration(
                output_dir='output_frames',
                num_frames=16,
                rows=2,
                cols=8,
                frame_type='flow',
                output_name='thesis_illustration_flow.jpg'
            )
            
            # Tạo ảnh minh họa từ frame gốc (chưa có optical flow)
            print("\nĐang tạo ảnh minh họa từ frame gốc (chưa có optical flow)...")
            create_thesis_illustration(
                output_dir='output_frames',
                num_frames=16,
                rows=2,
                cols=8,
                frame_type='original',
                output_name='thesis_illustration_original.jpg'
            )
            
            print("\n" + "=" * 60)
            print("HOÀN THÀNH!")
            print("=" * 60)
            print(f"\nĐã xử lý {total_frames} frames")
            print("Kết quả được lưu trong thư mục: output_frames/")
            print("\nCác file quan trọng:")
            print("  - thesis_illustration.jpg (16 frames overlay)")
            print("  - thesis_illustration_flow.jpg (16 frames flow)")
            print("  - thesis_illustration_original.jpg (16 frames gốc, chưa có optical flow)")
            print("  - frame_XXXX_original.jpg (từng frame gốc)")
            print("  - frame_XXXX_overlay.jpg (từng frame overlay)")
            print("  - frame_XXXX_flow.jpg (từng frame flow)")
