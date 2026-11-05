#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RK3568 量化部署脚本（rknn-toolkit2）。

⚠️ 运行前提
------------------------------------------------------------------
rknn-toolkit2 **只能安装在 Linux x86_64**（官方不提供 Windows 版本）。
请在 WSL2 / Ubuntu 22.04 / 服务器上执行，例如：

    pip install rknn-toolkit2==1.6.0 -i https://mirrors.aliyun.com/pypi/simple
    python deploy/rknn_quant.py --onnx exported/rtmpose-m.onnx
    python deploy/rknn_quant.py --onnx exported/rtmpose-m.onnx --mode hybrid

本脚本修正了常见示例里最容易踩的两个坑：
1. 忘了先调用 `rknn.config(...)`（缺少它 load_onnx 会直接报错）；
2. mean/std 没按 RTMPose 的归一化参数填写，导致量化校准分布完全错误。
"""

import argparse
import os

# RTMPose 在 MMPose 中的归一化参数（0-255 尺度，RGB 顺序）
# 对应 Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
RTMPOSE_MEAN = [[123.675, 116.28, 103.53]]
RTMPOSE_STD = [[58.395, 57.12, 57.375]]


def parse_args():
    parser = argparse.ArgumentParser(description="RK3568 模型量化")
    parser.add_argument("--onnx", required=True, help="输入 ONNX 模型路径")
    parser.add_argument("--out", default=None,
                        help="输出 .rknn 路径，默认与 ONNX 同名")
    parser.add_argument("--dataset", default="deploy/calib.txt",
                        help="校准集列表文件，每行一张已按模型输入尺寸"
                             "预处理好的图片路径")
    parser.add_argument("--platform", default="rk3568",
                        choices=["rk3568", "rk3576", "rk3588"],
                        help="目标芯片，默认 rk3568")
    parser.add_argument("--mode", default="ptq", choices=["ptq", "hybrid"],
                        help="ptq=全 int8 量化；hybrid=混合量化（保留敏感层"
                             "为 float16，精度更好、速度略慢）")
    parser.add_argument("--do-quant", type=int, default=1,
                        help="1=量化，0=仅转换（用于对比 float 精度）")
    parser.add_argument("--step", type=int, default=1, choices=[1, 2],
                        help="混合量化阶段：1=生成配置，2=按修改后的配置生成")
    parser.add_argument("--analysis", action="store_true",
                        help="量化后做逐层精度分析，输出各层误差")
    return parser.parse_args()


def make_rknn(verbose=True):
    """兼容 rknn-toolkit2 1.x / 2.x 的构造方式。"""
    from rknn.api import RKNN

    try:
        return RKNN(verbose=verbose)   # toolkit2 >= 1.6 常见签名
    except TypeError:
        return RKNN()                  # 旧版本不接受 verbose 参数


def main():
    args = parse_args()

    if args.out is None:
        base = os.path.splitext(os.path.basename(args.onnx))[0]
        args.out = os.path.join(os.path.dirname(args.onnx), f"{base}.rknn")

    has_dataset = os.path.isfile(args.dataset)
    if args.do_quant and not has_dataset:
        raise FileNotFoundError(
            f"量化需要校准集，但未找到 {args.dataset}。"
            f"请先用 deploy/make_calib.py 生成，或加 --do-quant 0 只做转换。")

    rknn = make_rknn()

    print(f"[1/5] config -> target_platform={args.platform}")
    rknn.config(
        mean_values=RTMPOSE_MEAN,
        std_values=RTMPOSE_STD,
        target_platform=args.platform,
        quantized_dtype="asymmetric_quantized-8",
        quantized_algorithm="normal",
        optimization_level=3,
    )

    print(f"[2/5] load_onnx -> {args.onnx}")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        raise RuntimeError(f"load_onnx 失败，错误码 {ret}")

    print(f"[3/5] build -> do_quantization={bool(args.do_quant)}")
    if args.mode == "hybrid":
        # 混合量化：step1 产出量化配置文件与中间模型，
        # 手动把 SimCC head 等敏感层改回 float16 后再 step2。
        print("      混合量化 step1：生成 ./rtmpose.quantization.cfg")
        ret = rknn.hybrid_quantization_step1(dataset=args.dataset)
        if ret != 0:
            raise RuntimeError(f"hybrid_quantization_step1 失败，错误码 {ret}")
        print("\n>>> 请打开生成的 *.quantization.cfg，"
              "把输出层/head 相关层的 quantized_type 改为 float16，"
              "保存后重新执行：")
        print(f"    python deploy/rknn_quant.py --onnx {args.onnx} "
              f"--mode hybrid --step 2\n")
        if args.step != 2:
            rknn.release()
            return
        ret = rknn.hybrid_quantization_step2(
            model_input="./rtmpose.model",
            data_input="./rtmpose.data",
            output_sequences=None,
        )
        if ret != 0:
            raise RuntimeError(f"hybrid_quantization_step2 失败，错误码 {ret}")
    else:
        ret = rknn.build(do_quantization=bool(args.do_quant),
                         dataset=args.dataset if args.do_quant else None)
        if ret != 0:
            raise RuntimeError(f"build 失败，错误码 {ret}")

    print(f"[4/5] export_rknn -> {args.out}")
    ret = rknn.export_rknn(args.out)
    if ret != 0:
        raise RuntimeError(f"export_rknn 失败，错误码 {ret}")

    if args.analysis:
        print("[5/5] accuracy_analysis（定位精度损失最大的层）")
        with open(args.dataset, "r", encoding="utf-8") as f:
            calib_imgs = [ln.strip() for ln in f if ln.strip()]
        rknn.accuracy_analysis(inputs=calib_imgs[:20],
                               output_dir="./analysis",
                               target=args.platform)

    print("[5/5] release")
    rknn.release()
    print(f"\n完成：{args.out}")
    print("提示：板端实测帧率请用 rknn-toolkit-lite2 在 RK3568 上跑，"
          "x86 模拟器上的耗时不能代表真实性能。")


if __name__ == "__main__":
    main()
