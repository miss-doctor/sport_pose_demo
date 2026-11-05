#!/usr/bin/env bash
# 下载本项目需要的官方预训练权重
set -e
cd "$(dirname "$0")/.."
mkdir -p checkpoints

echo "下载 YOLOX-s (MMDet, COCO person) ..."
curl -L -o checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth \
    "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"

echo "下载 RTMPose-m (MMPose, COCO, 256x192) ..."
curl -L -o checkpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth \
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"

echo "完成。权重已保存到 checkpoints/"
