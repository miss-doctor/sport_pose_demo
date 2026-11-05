@echo off
rem Download official pretrained weights for Windows (curl or PowerShell required)
setlocal
pushd "%~dp0\.."
mkdir checkpoints 2>nul

echo 下载 YOLOX-s ...
curl -L -o checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth ^
    "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"

echo 下载 RTMPose-m ...
curl -L -o checkpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth ^
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"

echo 完成。
pause
