#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 MMEngine Runner 启动 RTMPose 训练 / 微调。

与网上流传的 `from mmpose.apis import train_model` 写法不同 ——
MMPose 1.x 已经移除了 train_model，官方入口是 MMEngine 的 Runner。

用法：
    # 用 COCO 预训练权重在自己的体育数据集上微调
    python tools/train_pose.py configs/sport_rtmpose_finetune.py --work-dir work_dirs/sport

    # 断点续训
    python tools/train_pose.py configs/sport_rtmpose_finetune.py --resume

    # 覆盖配置里的任意字段
    python tools/train_pose.py configs/sport_rtmpose_finetune.py \
        --cfg-options train_cfg.max_epochs=100 optim_wrapper.optimizer.lr=2e-4
"""

import argparse
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="训练 / 微调 RTMPose")
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("--work-dir", default=None, help="工作目录")
    parser.add_argument("--resume", action="store_true",
                        help="从 work_dir 里最新的权重断点续训")
    parser.add_argument("--amp", action="store_true",
                        help="开启自动混合精度（需要 GPU）")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction,
                        help="以 key=value 形式覆盖配置")
    return parser.parse_args()


def main():
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmpose.utils import register_all_modules

    args = parse_args()

    # 把项目根目录加入 sys.path，使自定义增强模块可被 custom_imports 导入
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"          # 单卡 / CPU 训练
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = os.path.join(
            "work_dirs", os.path.splitext(os.path.basename(args.config))[0])

    if args.amp:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.resume:
        cfg.resume = True

    print(f"工作目录: {cfg.work_dir}")
    runner = Runner.from_cfg(cfg)
    runner.train()


class DictAction(argparse.Action):
    """把 key=value 列表解析为字典（与 MMEngine 内置行为一致）。"""

    def __call__(self, parser, namespace, values, option_string=None):
        out = getattr(namespace, self.dest, None) or {}
        for value in values:
            key, val = value.split("=", 1)
            try:
                parsed = yaml.safe_load(val)
            except Exception:
                parsed = val
            cur = out
            keys = key.split(".")
            for k in keys[:-1]:
                cur = cur.setdefault(k, {})
            cur[keys[-1]] = parsed
        setattr(namespace, self.dest, out)


if __name__ == "__main__":
    main()
