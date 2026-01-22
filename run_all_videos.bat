@echo off
REM Script batch để chạy xử lý tất cả 12 video
REM Kích hoạt môi trường ảo nếu có

echo ========================================
echo XU LY TAT CA 12 VIDEO
echo ========================================
echo.

REM Kiểm tra và kích hoạt môi trường ảo nếu có
if exist myenv\Scripts\activate.bat (
    echo Dang kich hoat moi truong ao...
    call myenv\Scripts\activate.bat
)

REM Chạy script Python
echo Dang chay script xu ly...
python process_all_videos.py

pause
