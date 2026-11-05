#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频 / 摄像头的两阶段姿态推理，输出带骨架的视频并记录帧率。

用法：
    python demo/video_demo.py --input data/videos/sport.mp4
    python demo/video_demo.py --input 0            # 摄像头
    python demo/video_demo.py --input demo.mp4 --no-save --show
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
    parser = argparse.ArgumentParser(description="视频/摄像头人体姿态推理")
    parser.add_argument("--input", required=True,
                        help="视频路径，或摄像头序号（如 0）")
    parser.add_argument("--out-file", default=None,
                        help="输出视频路径，默认 results/<name>_pose.mp4")
    parser.add_argument("--det-config", default=DEFAULT_DET_CONFIG)
    parser.add_argument("--det-ckpt", default=DEFAULT_DET_CKPT)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--pose-ckpt", default=DEFAULT_POSE_CKPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bbox-thr", type=float, default=0.5)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="每隔 N 帧处理 1 帧（0 表示逐帧）")
    parser.add_argument("--no-angles", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def open_source(src):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {src}")
    return cap


def main():
    args = parse_args()

    print("初始化模型 ...")
    pipeline = SportPosePipeline(args.det_config, args.det_ckpt,
                                 args.pose_config, args.pose_ckpt,
                                 device=args.device, bbox_thr=args.bbox_thr,
                                 kpt_thr=args.kpt_thr)

    cap = open_source(args.input)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    print(f"视频源: {args.input}  {width}x{height} @ {fps_in:.1f} FPS, "
          f"共 {total_frames if total_frames > 0 else '未知'} 帧")

    writer = None
    if not args.no_save:
        out = args.out_file
        if out is None:
            os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
            name = (os.path.splitext(os.path.basename(args.input))[0]
                    if not args.input.isdigit() else "webcam")
            out = os.path.join(ROOT, "results", f"{name}_pose.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out, fourcc, fps_in, (width, height))
        print(f"输出视频: {out}")

    frame_idx = 0
    processed = 0
    total_ms = 0.0
    step = max(1, args.skip_frames + 1)
    last_canvas = None
    print("开始处理（按 Q 或 Ctrl+C 结束）...")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if (frame_idx - 1) % step != 0:
                if last_canvas is not None and writer is not None:
                    writer.write(last_canvas)
                continue

            t0 = time.perf_counter()
            results = pipeline(frame, with_angles=not args.no_angles)
            elapsed = (time.perf_counter() - t0) * 1000
            total_ms += elapsed
            processed += 1

            canvas = pipeline.visualize(frame, results,
                                        draw_angles=not args.no_angles)
            cv2.putText(canvas, f"{elapsed:.0f} ms | {len(results)} person(s)",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40),
                        3, cv2.LINE_AA)
            cv2.putText(canvas, f"{elapsed:.0f} ms | {len(results)} person(s)",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
            last_canvas = canvas

            if writer is not None:
                writer.write(canvas)
            if args.show:
                cv2.imshow("SportPose", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n已中断")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if processed:
        avg = total_ms / processed
        print(f"\n处理 {processed} 帧 | 平均 {avg:.1f} ms/帧 "
              f"({1000.0 / avg:.1f} FPS) | device={args.device}")
        print("提示：CPU 上的帧率仅供参考，RK3568 等边缘设备的实际帧率 "
              "需在板端用 rknn 推理实测。")


if __name__ == "__main__":
    main()
