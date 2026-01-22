#!/bin/bash
# Script bash để chạy xử lý tất cả 12 video
# Kích hoạt môi trường ảo nếu có

echo "========================================"
echo "XU LY TAT CA 12 VIDEO"
echo "========================================"
echo ""

# Kiểm tra và kích hoạt môi trường ảo nếu có
if [ -f "myenv/bin/activate" ]; then
    echo "Dang kich hoat moi truong ao..."
    source myenv/bin/activate
fi

# Chạy script Python
echo "Dang chay script xu ly..."
python process_all_videos.py
