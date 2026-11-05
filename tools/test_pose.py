#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在验证集上得到真实、可复现的姿态指标（COCO OKS-based AP/AR）。

网上常说的 "OKS = xx%" 通常指的就是 COCO 评估里的 AP（因为 COCO AP
是按 OKS 阈值 0.5:0.95 平均出来的）。
运行本脚本你会看到：
    AP    : 0.746   ← 对应常说的 "OKS AP"
    AP_50 : 0.899   ← OKS@0.50
    AP_75 : 0.817   ← OKS@0.75
    AR    : 0.795

用法：
    # 官方 COCO 预训练权重（需提前下载 COCO val）
    python tools/test_pose.py configs/pose/rtmpose-m_8xb256-420e_coco-256x192.py

    # 自己微调后的权重
    python tools/test_pose.py configs/sport_rtmpose_finetune.py \
        --checkpoint work_dirs/sport/best_coco_AP_epoch_xx.pth
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="评估 RTMPose 姿态指标")
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("--checkpoint", default=None,
                        help="权重路径；不填则使用 cfg.load_from 或 cfg.resume")
    parser.add_argument("--work-dir", default=None)
    return parser.parse_args()


def main():
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmpose.utils import register_all_modules

    args = parse_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.checkpoint is not None:
        cfg.load_from = args.checkpoint

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    print("\n==== 评估结果 ====")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
