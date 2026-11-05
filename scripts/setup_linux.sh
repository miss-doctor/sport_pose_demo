#!/usr/bin/env bash
# Linux（CUDA 11.8/12.1 GPU）推荐安装方式
# 运行前：conda create -n sport_pose python=3.11 && conda activate sport_pose

set -e

cd "$(dirname "$0")/.."

PYTHON=python

echo "[1/6] torch（按需求改 cu118 / cu121 / cpu）..."
$PYTHON -m pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[2/6] 基础依赖 ..."
$PYTHON -m pip install "numpy<2" opencv-python matplotlib pyyaml

echo "[3/6] MMEngine / OpenMIM ..."
$PYTHON -m pip install -U openmim mmengine==0.10.7

echo "[4/6] mmcv（mim 会自动选对应 torch/cuda/python 的预编译轮子）..."
mim install "mmcv==2.1.0"

echo "[5/6] MMDet / MMPose ..."
$PYTHON -m pip install mmdet==3.3.0
$PYTHON -m pip install mmpose==1.3.2

echo "[6/6] 可选：ONNX 导出 / RKNN 量化依赖 ..."
$PYTHON -m pip install onnx onnxruntime
# rknn-toolkit2 只能在 x86 Linux 安装：
# $PYTHON -m pip install rknn-toolkit2==1.6.0

echo "完成。运行 bash scripts/download_models.sh 下载权重。"
