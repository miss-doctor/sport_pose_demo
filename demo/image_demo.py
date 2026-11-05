#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单张图片的 MMDet(YOLO) + MMPose(RTMPose) 两阶段姿态推理示例。

用法：
    python demo/image_demo.py --input data/images/demo.jpg
    python demo/image_demo.py --input a.jpg --out-file out.jpg --show
"""

import argparse
import os
import sys
import time

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sport_pose.infer import SportPosePipeline  # noqa: E402

DEFAULT_DET_CONFIG = os.path.join(
    ROOT, "configs", "det", "yolox_s_8xb8-300e_coco.py")
DEFAULT_DET_CKPT = os.path.join(
    ROOT, "checkpoints",
    "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth")
DEFAULT_POSE_CONFIG = os.path.join(
    ROOT, "configs", "pose", "rtmpose-m_8xb256-420e_coco-256x192.py")
DEFAULT_POSE_CKPT = os.path.join(
    ROOT, "checkpoints",
    "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="单图人体姿态推理")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--out-file", default=None,
                        help="结果图输出路径，默认 results/xxx_pose.jpg")
    parser.add_argument("--det-config", default=DEFAULT_DET_CONFIG)
    parser.add_argument("--det-ckpt", default=DEFAULT_DET_CKPT)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--pose-ckpt", default=DEFAULT_POSE_CKPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bbox-thr", type=float, default=0.5)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--no-angles", action="store_true",
                        help="不计算/标注关节角度")
    parser.add_argument("--show", action="store_true", help="弹窗显示结果")
    return parser.parse_args()


def print_result(results, kpt_thr=0.3):
    for i, item in enumerate(results):
        x1, y1, x2, y2 = item["bbox"]
        print(f"[人体 {i + 1}] bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        visible = int((item["scores"] >= kpt_thr).sum())
        print(f"         可见关键点 {visible}/17")
        if "angles" in item:
            parts = []
            for name in ("left_elbow", "right_elbow", "left_knee",
                         "right_knee", "torso_tilt"):
                val = item["angles"].get(name)
                parts.append(f"{name}={'--' if val is None else f'{val:.1f}'}")
            print("         关节角: " + "  ".join(parts))


def main():
    args = parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {args.input}")

    print("初始化模型 ...")
    pipeline = SportPosePipeline(args.det_config, args.det_ckpt,
                                 args.pose_config, args.pose_ckpt,
                                 device=args.device, bbox_thr=args.bbox_thr,
                                 kpt_thr=args.kpt_thr)
    print(f"  检测器: {os.path.basename(args.det_config)}")
    print(f"  姿态模型: {os.path.basename(args.pose_config)}")

    # 预热一次，避免把首次推理的算子初始化时间计入耗时
    pipeline(img.copy())

    t0 = time.perf_counter()
    results = pipeline(img, with_angles=not args.no_angles)
    total = time.perf_counter() - t0

    print(f"\n检测到 {len(results)} 个人体")
    print_result(results, args.kpt_thr)
    print(f"\n耗时: 检测 {pipeline.det_time * 1000:.1f} ms | "
          f"姿态 {pipeline.pose_time * 1000:.1f} ms | "
          f"合计 {total * 1000:.1f} ms "
          f"({1.0 / max(total, 1e-6):.1f} FPS, device={args.device})")

    canvas = pipeline.visualize(img, results, draw_angles=not args.no_angles)

    out = args.out_file
    if out is None:
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        out = os.path.join(ROOT, "results", f"{base}_pose.jpg")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cv2.imwrite(out, canvas)
    print(f"\n结果已保存: {out}")

    if args.show:
        cv2.imshow("SportPose", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
