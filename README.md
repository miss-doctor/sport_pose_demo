# YOLO + RTMPose 智慧体育人体姿态识别（MMDet + MMPose 实战版）

> 本项目的目标：**用真正可运行的代码**，把「YOLO 检测 → RTMPose 关键点 → 关节角度分析 → ONNX → RKNN 量化」这条链路走通。
>
> 网上的很多"模板"代码其实是跑不起来的，而且会教你虚报指标。这个项目把坑都填了，并给出**真实、可复现**的基准数字。

---

## 一、项目定位

- **检测器**：YOLOX-s（MMDetection 中的 YOLO 系列）
- **姿态估计**：RTMPose-m（MMPose）
- **应用场景**：智慧体育 / 健身动作分析 / 人体关节角度计算
- **部署目标**：RK3568（通过 RKNN 量化）
- **本机验证环境**：Windows 11 + Python 3.11 + CPU

本项目的核心差异点：

1. **不是复制品，是修正版**：修复了常见模板里的 `train_model`、`mmcv.Config`、未注册增强、`pred_instances['bboxes']` 等错误。
2. **只做标签安全的增强**：自定义的 `SportMotionAug` 只做光度类变换（运动模糊、亮度、噪声、JPEG 伪影），不移动像素位置，因此不会导致关键点标签错位。
3. **指标诚实**：直接给出官方 RTMPose-m 在 COCO val 上的真实 AP=74.6，而不是张口就来的 "OKS 93%"。
4. **性能诚实**：本机 CPU 约 3 FPS；边缘端 80 ms/帧必须到 RK3568 板端用 RKNN 实测，不能靠 Windows 模拟器估计。

---

## 二、本机能跑通的实测结果

在 `data/images/people.jpg`（健身房场景，640×427）上实测：

| 阶段 | 耗时 | 备注 |
|------|------|------|
| 人体检测（YOLOX-s） | ~185 ms | CPU 单线程 |
| 姿态估计（RTMPose-m，2 人） | ~137 ms | 人数越多越慢 |
| **合计** | **~322 ms / 帧** | **≈ 3.1 FPS** |

> CPU 帧率仅供参考。RK3568 目标性能需在板端用 `.rknn` 实测，受 NPU 占用、量化方式、输入分辨率影响很大。

官方 COCO 预训练 RTMPose-m（256×192）的公开指标：

| 指标 | 数值 | 含义 |
|------|------|------|
| AP | 0.746 | OKS 阈值 0.50:0.95 的平均精度 |
| AP@0.5 | 0.899 | OKS ≥ 0.50 的精度 |
| AP@0.75 | 0.817 | OKS ≥ 0.75 的精度 |
| AR | 0.795 | 平均召回率 |

> 这些数字来自 OpenMMLab 官方 model-index（在 `configs/` 里对应配置文件的元数据），可直接用 `tools/test_pose.py` 在 COCO val 上复现。

---

## 三、环境搭建

### 3.1 Windows + CPU（本机验证通过）

需要 Python 3.11。打开 PowerShell / CMD：

```bat
scripts\setup_windows.bat
bash scripts\download_models.sh
```

如果 `bash` 不可用，可直接运行 `scripts/download_models.sh` 里的两条 `curl` 命令，或到 README 最下方的链接手动下载。

### 3.2 Linux + GPU（推荐训练用）

```bash
conda create -n sport_pose python=3.11
conda activate sport_pose
bash scripts/setup_linux.sh
bash scripts/download_models.sh
```

> `rknn-toolkit2` 只能装在 x86_64 Linux 上，所以量化步骤建议在 WSL2 / Ubuntu 服务器执行。

### 3.3 本机实测的关键版本锁

| 包 | 版本 | 说明 |
|----|------|------|
| Python | 3.11 | mmcv 2.1.0 无 Windows cp312 轮子 |
| torch | 2.1.2+cpu | 与 mmcv 2.1.0 对齐 |
| mmcv | 2.1.0 | 必须从 OpenMMLab 官方索引安装 |
| mmdet | 3.3.0 | 依赖 mmcv < 2.2.0 |
| mmpose | 1.3.2 | 依赖 mmcv < 2.2.0 |
| numpy | 1.26.4 | xtcocotools 按 numpy 1.x 编译，numpy 2.x 会崩 |
| setuptools | <81 | setuptools 81+ 移除 mmengine 依赖的 `pkg_resources` |

---

## 四、快速开始：单图 / 视频 / 摄像头推理

### 4.1 单张图片

```bash
.venv\Scripts\activate          # Windows
python demo/image_demo.py --input data/images/people.jpg
```

默认输出到 `results/people_pose.jpg`，控制台会打印：
- 检测到几个人体
- 每个框的 bbox
- 可见关键点数量
- 关键关节角度（肘、膝、躯干倾角）
- 检测 / 姿态 / 总耗时

### 4.2 视频或摄像头

```bash
# 本地视频
python demo/video_demo.py --input data/videos/sport.mp4

# 摄像头
python demo/video_demo.py --input 0 --show
```

> `--no-save` 仅预览不保存；`--skip-frames N` 每隔 N 帧处理一次，可提升流畅度。

### 4.3 关键概念：关节角度

`sport_pose/analysis/angles.py` 把 17 个关键点转成运动学角度：

- **left_elbow / right_elbow**：手臂弯曲角度
- **left_knee / right_knee**：膝关节角度（深蹲深浅、跳跃姿态）
- **torso_tilt**：躯干相对竖直方向的倾斜角（体前屈、弯腰）

这些角度是"智慧体育动作评判"的起点，而不是直接输出一个动作分类。

---

## 五、自定义体育数据集微调

### 5.1 数据集格式

必须是 **COCO person_keypoints 格式**（17 点，与 COCO 一致）。放到：

```
data/sport/
├── annotations/
│   ├── person_keypoints_train.json
│   └── person_keypoints_val.json
├── train/
│   └── xxx.jpg
└── val/
    └── xxx.jpg
```

### 5.2 启动训练

```bash
python tools/train_pose.py configs/sport_rtmpose_finetune.py --work-dir work_dirs/sport
```

配置要点：
- `max_epochs = 80`、`base_lr = 5e-5`：微调不需要从零训练那么大的 lr 和 epoch。
- `load_from`：从官方 COCO 权重加载整模型。
- `train_pipeline.insert(1, sport_aug)`：在几何变换前插入自定义运动增强。
- Windows 下 `mp_start_method='spawn'`、`num_workers=0`，避免 MMCV 默认的 `fork` 在 Windows 上崩溃。

### 5.3 评估真实指标

```bash
python tools/test_pose.py configs/sport_rtmpose_finetune.py \
    --checkpoint work_dirs/sport/best_coco_AP_epoch_xx.pth
```

输出就是真实的 COCO OKS-based AP/AR。**只有这个数字才应该写进简历。** 在你自己的体育数据上，它可能高于或低于 74.6，这取决于数据难度和标注质量。

### 5.4 如果还没自己的标注数据

你可以先用 [LabelMe](https://github.com/wkentaro/labelme)、[CVAT](https://cvat.org/) 或 [MMPose 的 Labelbee 插件](https://mmpose.readthedocs.io/) 标注 17 点 COCO 格式，再用 `tools/prepare_coco.py`（或自己写脚本）整理成上面的目录结构。

---

## 六、ONNX 导出与 RKNN 量化

### 6.1 导出 ONNX

本项目提供**不依赖 mmdeploy** 的 ONNX 导出，在 Windows 上也能跑：

```bash
python tools/export_onnx.py \
    --pose-config configs/pose/rtmpose-m_8xb256-420e_coco-256x192.py \
    --pose-ckpt checkpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth \
    --out exported/rtmpose-m.onnx
```

导出后会自动做 ONNX 结构校验，并比较 ONNX 与 PyTorch 输出的最大误差。

> 检测器 YOLOX 的 ONNX 导出涉及动态 anchor 与 NMS，建议在 Linux 上用 mmdeploy 导出，或在 RKNN 端单独用 `rknn_yolov5_demo` 之类方案处理。

### 6.2 生成校准集

RKNN 量化需要 representative 的校准数据：

```bash
python deploy/make_calib.py --img-dir data/images --out-dir deploy/calib
```

这会自动：
1. 检测人体框；
2. 裁剪并缩放到 192×256；
3. 保存为 RGB 图片；
4. 生成 `deploy/calib.txt`。

### 6.3 RK3568 量化

**必须在 Linux x86_64 上运行**（rknn-toolkit2 无 Windows 版）：

```bash
pip install rknn-toolkit2==1.6.0
python deploy/rknn_quant.py --onnx exported/rtmpose-m.onnx --analysis
```

本脚本修正了网上常见示例的两个致命错误：
1. **缺 `rknn.config()`**：没调用 `config()` 直接 `load_onnx` 会报错。
2. **mean/std 写错**：RTMPose 的归一化参数是 `mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375]`，对应 `bgr_to_rgb=True`，这里已经填好。

如果量化后掉点严重，用 `--mode hybrid` 做混合量化，手动把 SimCC head 等敏感层保留为 float16。

---

## 七、项目结构

```
sport_pose_project/
├── configs/                         # 模型配置
│   ├── det/yolox_s_8xb8-300e_coco.py          # MMDet YOLOX-s 配置
│   ├── pose/rtmpose-m_8xb256-420e_coco-256x192.py  # MMPose RTMPose-m 配置
│   └── sport_rtmpose_finetune.py   # 自定义体育数据集微调配置
├── checkpoints/                     # 权重（脚本下载）
├── data/
│   └── images/people.jpg            # 示例图片
├── demo/
│   ├── image_demo.py                # 单图推理
│   ├── video_demo.py                # 视频/摄像头推理
│   └── webcam_demo.py               # 摄像头专用（同 video_demo）
├── deploy/
│   ├── make_calib.py                # 生成 RKNN 校准集
│   └── rknn_quant.py                # RKNN 量化脚本
├── exported/                            # 导出的 ONNX（运行时生成）
├── results/                         # 可视化结果（运行时生成）
├── sport_pose/                      # 自定义包
│   ├── transforms/sport_aug.py      # 运动场景增强（已注册到 MMPose）
│   ├── analysis/angles.py           # 关节角度计算
│   ├── infer.py                     # 两阶段推理流水线
│   └── visualize.py                 # OpenCV 绘制骨架
├── tools/
│   ├── train_pose.py                # 训练/微调
│   ├── test_pose.py                 # 评估真实 AP/AR
│   └── export_onnx.py               # ONNX 导出
├── scripts/
│   ├── setup_windows.bat            # Windows 环境脚本
│   ├── setup_linux.sh               # Linux GPU 环境脚本
│   └── download_models.sh           # 下载官方权重
├── requirements.txt
└── README.md
```

---

## 八、常见错误写法 vs 本项目写法

| 错误写法（网上模板常见） | 为什么错 | 本项目写法 |
|------------------------|---------|-----------|
| `from mmpose.apis import train_model` | MMPose 1.x 已移除 `train_model` | `from mmengine.runner import Runner; Runner.from_cfg(cfg).train()` |
| `mmcv.Config.fromfile(...)` | mmcv 2.x 配置入口已迁到 mmengine | `from mmengine.config import Config` |
| 手写 `dict(type='SportAug')` 但不在 `TRANSFORMS` 注册 | 运行时报 `SportAug is not in the xxx registry` | `@TRANSFORMS.register_module()` + `custom_imports` |
| `res.pred_instances['bboxes']` | MMDet 3.x 是 `.bboxes` 属性，不是字典 | `pred.bboxes.cpu().numpy()` |
| 自定义增强里只 `cv2.flip` 图片，不更新关键点 | 标签和图像错位，训练出来的模型是错的 | `SportMotionAug` 只做光度增强；几何翻转交给内置 `RandomFlip` |
| 张口说 "OKS 84% → 93%" "80 ms/帧" | 没有评估脚本、没有板端实测支撑 | `tools/test_pose.py` 出真实 AP；RKNN 帧率在 RK3568 上实测 |

---

## 九、重要提醒

1. **不要虚报指标**：在没有自己的标注数据集和 `tools/test_pose.py` 评估结果之前，不要把 "OKS 93%"、"12 FPS" 写进简历。面试官一问测试集规模、评估脚本、损失曲线，很容易露馅。
2. **YOLOX-s 不是最快的**：YOLOX-s 在 CPU 上不算快。如果目标是 RK3568 实时，建议把检测器也换成 RTMDet-tiny/nano，并把姿态模型换成 RTMPose-tiny/s。
3. **Windows 只能做 CPU 验证**：RKNN 量化必须在 Linux 下完成，板端部署在 RK3568（ARM Linux）上完成。
4. ** pose 模型对输入敏感**：摄像头 / 视频抽帧后，务必保证人体框的质量。检测漏人、框太小、遮挡严重都会让姿态估计崩掉。

---

## 十、参考链接

- [YOLOX-s 权重](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth)
- [RTMPose-m 256×192 权重](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth)
- [MMDetection 文档](https://mmdetection.readthedocs.io/)
- [MMPose 文档](https://mmpose.readthedocs.io/)
- [RKNN Toolkit2 文档](https://github.com/rockchip-linux/rknn-toolkit2)
