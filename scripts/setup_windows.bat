@echo off

setlocal EnableDelayedExpansion
pushd "%~dp0\.."

echo [1/7] 创建虚拟环境 ...
python -m venv .venv

if not exist ".venv\Scripts\activate.bat" (
    echo 虚拟环境创建失败，请检查 python 是否在 PATH 中。
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo [2/7] 升级 pip / setuptools（setuptools 81+ 会缺少 pkg_resources）...
python -m pip install -U "pip<25" "setuptools<81" wheel

echo [3/7] 安装 torch CPU 版 ...
python -m pip install torch==2.1.2+cpu torchvision==0.16.2+cpu --index-url https://download.pytorch.org/whl/cpu

echo [4/7] 安装基础视觉库 ...
python -m pip install "numpy<2" opencv-python matplotlib pyyaml

echo [5/7] 安装 MMEngine ...
python -m pip install mmengine==0.10.7

echo [6/7] 安装 mmcv（Windows 预编译轮子）...
python -m pip install https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/mmcv-2.1.0-cp311-cp311-win_amd64.whl

echo [7/7] 安装 MMDet 与 MMPose ...
python -m pip install mmdet==3.3.0
python -m pip install --no-build-isolation mmpose==1.3.2

echo.
echo 环境安装完成。接下来下载权重：
echo   bash scripts/download_models.sh
echo   或手动从 README 复制链接下载到 checkpoints/ 目录。
echo.
pause
