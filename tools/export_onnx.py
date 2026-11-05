#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 RTMPose（以及可选的检测器）导出为 ONNX，供 RKNN 量化使用。

导出姿态模型有两条路径：
1. mmdeploy（官方推荐）：自动处理前后处理拼接，需要 Linux 环境安装 mmdeploy。
2. 本脚本内置的后备路径：直接用 torch.onnx.export 导出
   「backbone + SimCC head」的纯网络部分，输出两个 SimCC 分支。
   这条路径在 Windows / CPU 上也能跑通，便于先拿到 .onnx 再做量化。

用法：
    # 后备路径（无需 mmdeploy）
    python tools/export_onnx.py --pose-config configs/pose/rtmpose-m_8xb256-420e_coco-256x192.py ^
        --pose-ckpt checkpoints/rtmpose-m_....pth --out exported/rtmpose-m.onnx

    # mmdeploy 路径
    python tools/export_onnx.py --use-mmdeploy --pose-config ... --pose-ckpt ...
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class PoseDeployModel(nn.Module):
    """只保留网络前向：输入归一化后的图像，输出 SimCC 的 x / y 两个分支。

    数据预处理（bbox 仿射变换、归一化）与后处理（SimCC 解码成 17 个坐标）
    在板端用 C/Python 实现，不进 ONNX。
    """

    def __init__(self, pose_model):
        super().__init__()
        self.backbone = pose_model.backbone
        self.head = pose_model.head

    def forward(self, x):
        return self.head(self.backbone(x))


def parse_args():
    parser = argparse.ArgumentParser(description="导出 ONNX")
    parser.add_argument("--pose-config",
                        default=os.path.join(
                            ROOT, "configs", "pose",
                            "rtmpose-m_8xb256-420e_coco-256x192.py"))
    parser.add_argument("--pose-ckpt", default=None,
                        help="姿态权重；不填则尝试从配置里继承 load_from")
    parser.add_argument("--out", default=os.path.join(ROOT, "exported",
                                                     "rtmpose-m.onnx"))
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--use-mmdeploy", action="store_true",
                        help="改用 mmdeploy 导出（需已安装 mmdeploy）")
    parser.add_argument("--det-config", default=None,
                        help="同时导出检测器时需提供 MMDet 配置")
    parser.add_argument("--det-ckpt", default=None)
    parser.add_argument("--det-out", default=os.path.join(ROOT, "onnx",
                                                          "yolox_s.onnx"))
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 ONNX 结构与数值校验")
    return parser.parse_args()


def export_by_mmdeploy(args):
    from mmdeploy.apis import torch2onnx

    print("使用 mmdeploy 导出 ...")
    torch2onnx(img=args.demo_img if hasattr(args, "demo_img") else "",
               work_dir=os.path.dirname(args.out),
               save_file=os.path.basename(args.out),
               deploy_cfg="",  # 由 mmdeploy 默认配置推断
               model_cfg=args.pose_config,
               model_checkpoint=args.pose_ckpt,
               device="cpu")


def export_manual(args):
    from mmpose.apis import init_model

    print("使用内置后备路径导出（torch.onnx.export）...")
    model = init_model(args.pose_config, args.pose_ckpt, device="cpu")
    model.eval()

    input_size = getattr(getattr(model, "codec", None), "input_size", None)
    if input_size is None:
        from mmengine.config import Config
        cfg = Config.fromfile(args.pose_config)
        input_size = cfg.codec.input_size
    # 注意：MMPose 的 input_size 是 (H, W)，不是 (W, H)
    h, w = int(input_size[0]), int(input_size[1])
    print(f"模型输入尺寸 (H, W) = ({h}, {w})")

    deploy_model = PoseDeployModel(model).eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dummy = torch.randn(1, 3, h, w)

    with torch.no_grad():
        torch.onnx.export(
            deploy_model,
            dummy,
            args.out,
            input_names=["input"],
            output_names=["simcc_x", "simcc_y"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    print(f"ONNX 已保存: {args.out}")

    if args.no_verify:
        return

    # ---- 结构校验 ----
    try:
        import onnx
        onnx_model = onnx.load(args.out)
        onnx.checker.check_model(onnx_model)
        print("ONNX 结构校验通过")
    except ImportError:
        print("未安装 onnx，跳过结构校验（pip install onnx）")
        return

    # ---- 数值一致性校验 ----
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("未安装 onnxruntime，跳过数值校验（pip install onnxruntime）")
        return

    with torch.no_grad():
        ref_x, ref_y = deploy_model(dummy)
    ref_x = ref_x.numpy()
    ref_y = ref_y.numpy()

    sess = ort.InferenceSession(args.out,
                                providers=["CPUExecutionProvider"])
    out_x, out_y = sess.run(None, {"input": dummy.numpy()})
    max_err = max(float(np.abs(ref_x - out_x).max()),
                  float(np.abs(ref_y - out_y).max()))
    print(f"ONNX 与 PyTorch 输出最大误差: {max_err:.3e}")
    if max_err > 1e-3:
        print("警告：误差偏大，建议检查 opset 版本与算子兼容性")


def export_detector(args):
    if not (args.det_config and args.det_ckpt):
        return
    try:
        from mmdeploy.apis import torch2onnx
    except ImportError:
        print("跳过检测器导出：MMDet 的 YOLO 头含动态 anchor 与 NMS，"
              "手工导出容易出错，请在 Linux 上用 mmdeploy 导出")
        return
    print("使用 mmdeploy 导出检测器 ...")
    torch2onnx(img=args.demo_img if hasattr(args, "demo_img") else "",
               work_dir=os.path.dirname(args.det_out),
               save_file=os.path.basename(args.det_out),
               deploy_cfg="",
               model_cfg=args.det_config,
               model_checkpoint=args.det_ckpt,
               device="cpu")
    print(f"检测器 ONNX 已保存: {args.det_out}")


def main():
    args = parse_args()
    if args.use_mmdeploy:
        export_by_mmdeploy(args)
    else:
        export_manual(args)
    export_detector(args)


if __name__ == "__main__":
    main()
