#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 RKNN 量化用的校准集（calib.txt + 预处理后的图片）。

校准集必须是**模型真正会看到的输入**：
1. 按检测框裁剪出来的人体（而不是整张原图，否则背景会污染量化分布）；
2. 缩放到姿态模型的输入尺寸（RTMPose-m 为 192x256，即 HxW）；
3. 转成 RGB 顺序（对应模型 bgr_to_rgb=True）。

用法：
    python deploy/make_calib.py --img-dir data/images --out-dir deploy/calib
"""

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sport_pose.infer import SportPosePipeline  # noqa: E402

# RTMPose-m 输入尺寸 (H, W)
INPUT_H, INPUT_W = 192, 256


def parse_args():
    parser = argparse.ArgumentParser(description="生成 RKNN 校准集")
    parser.add_argument("--img-dir", default="data/images",
                        help="原始图片目录")
    parser.add_argument("--out-dir", default="deploy/calib",
                        help="校准图片输出目录")
    parser.add_argument("--calib-list", default="deploy/calib.txt")
    parser.add_argument("--max-images", type=int, default=50,
                        help="最多处理多少张原图")
    parser.add_argument("--max-persons", type=int, default=200,
                        help="最多保留多少个裁剪样本")
    parser.add_argument("--det-config",
                        default=os.path.join(
                            ROOT, "configs", "det",
                            "yolox_s_8xb8-300e_coco.py"))
    parser.add_argument("--det-ckpt",
                        default=os.path.join(
                            ROOT, "checkpoints",
                            "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bbox-thr", type=float, default=0.5)
    return parser.parse_args()


def crop_person(img, bbox, padding=1.15):
    """按检测框带边距裁剪，并直接缩放到模型输入尺寸。

    说明：真正的 TopdownAffine 会按 3:4 宽高比重算框再做仿射变换，
    这里用「带边距裁剪 + 缩放」近似，对校准用途足够，且实现简单可靠。

    Args:
        img (np.ndarray): BGR 原图。
        bbox (np.ndarray): (4,) xyxy。
        padding (float): 边距放大倍数。默认 1.15。

    Returns:
        np.ndarray: RGB、尺寸为 (INPUT_H, INPUT_W) 的 uint8 图像。
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # 以 3:4 (H:W) 为目标比例，短边补齐
    target_ratio = INPUT_H / INPUT_W
    if w / max(h, 1e-6) > 1 / target_ratio:
        h = w * target_ratio
    else:
        w = h / target_ratio
    w *= padding
    h *= padding

    ih, iw = img.shape[:2]
    x1 = int(max(0, cx - w / 2))
    y1 = int(max(0, cy - h / 2))
    x2 = int(min(iw, cx + w / 2))
    y2 = int(min(ih, cy + h / 2))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(
        os.path.join(args.img_dir, f)
        for f in os.listdir(args.img_dir) if f.lower().endswith(exts)
    )[:args.max_images]
    if not files:
        raise FileNotFoundError(f"{args.img_dir} 下没有可用图片")

    print(f"初始化检测器，处理 {len(files)} 张图片 ...")
    # 只做检测，不加载姿态模型
    pipeline = SportPosePipeline(args.det_config, args.det_ckpt,
                                 None, None, device=args.device)
    del pipeline.pose_model  # 校准只需要检测器

    os.makedirs(args.out_dir, exist_ok=True)
    saved = []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        boxes = pipeline.detect(img)
        for i, box in enumerate(boxes):
            if len(saved) >= args.max_persons:
                break
            crop = crop_person(img, box)
            if crop is None:
                continue
            name = (f"{os.path.splitext(os.path.basename(path))[0]}"
                    f"_{i}.jpg")
            out_path = os.path.join(args.out_dir, name)
            cv2.imwrite(out_path, crop)
            saved.append(out_path)
        if len(saved) >= args.max_persons:
            break

    with open(args.calib_list, "w", encoding="utf-8") as f:
        for p in saved:
            f.write(os.path.abspath(p).replace("\\", "/") + "\n")

    print(f"生成 {len(saved)} 个校准样本 -> {args.out_dir}")
    print(f"校准列表 -> {args.calib_list}")
    print("提示：建议至少 20~50 张样本，且尽量来自真实运动场景。")


if __name__ == "__main__":
    main()
