# 在自制体育姿态数据集上微调 RTMPose-m

# 运行：
#   python tools/train_pose.py configs/sport_rtmpose_finetune.py --work-dir work_dirs/sport

_base_ = ['./pose/rtmpose-m_8xb256-420e_coco-256x192.py']

# 子配置中需要显式定义这些变量，才能在 dataloaders / pipelines 里引用。
# 值与基配置保持一致。
dataset_type = 'CocoDataset'
data_mode = 'topdown'

codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# ------------------------------------------------------------------
# 1. 让 MMPose 能找到自定义增强模块
#    （这一行不能省，否则会报 SportMotionAug is not in TRANSFORMS registry）
# ------------------------------------------------------------------
custom_imports = dict(imports=['sport_pose.transforms'],
                      allow_failed_imports=False)

# ------------------------------------------------------------------
# 2. 训练轮数与学习率
#    微调场景不需要 420 epoch，80 epoch 通常足够；
#    学习率也远小于从零训练的 4e-3。
# ------------------------------------------------------------------
max_epochs = 80
stage2_num_epochs = 20
base_lr = 5e-5

train_cfg = dict(max_epochs=max_epochs, val_interval=5)
randomness = dict(seed=21)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-3,
        by_epoch=False,
        begin=0,
        end=800),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# 微调时自己控制 lr，不让 auto_scale_lr 再按 1024 batch 缩放
auto_scale_lr = None

# 用官方 COCO 权重作为起点（整模型加载，不只是 backbone）
load_from = 'checkpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192' \
            '-d8dd5ca4_20230127.pth'

# ------------------------------------------------------------------
# 3. 运动场景增强
#    位置：紧跟 LoadImage（在处理几何变换之前做画质退化）
#    只做光度类变换，不会让关键点标签错位。
# ------------------------------------------------------------------
sport_aug = dict(
    type='SportMotionAug',
    motion_blur_prob=0.5,
    motion_blur_kernel_choices=(3, 5, 7, 9),
    directional_prob=0.6,
    brightness_range=(0.7, 1.3),
    contrast_range=(0.75, 1.25),
    brightness_prob=0.6,
    noise_std_range=(0, 10),
    noise_prob=0.3,
    jpeg_quality_range=(60, 95),
    jpeg_prob=0.3)

# 子配置中不能直接引用 base 定义的 train_pipeline 变量，
# 因此显式重写 pipeline，把 sport_aug 插在 LoadImage 之后。
train_pipeline = [
    dict(type='LoadImage'),
    sport_aug,
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage'),
    sport_aug,
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs')
]

# ------------------------------------------------------------------
# 4. 数据集路径（换成自己的体育数据）
# ------------------------------------------------------------------
data_root = 'data/sport/'

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val.json')
test_evaluator = val_evaluator

# ------------------------------------------------------------------
# 5. 跨平台兼容
#    mmpose 默认 mp_start_method='fork'，Windows 不支持，必须改成 spawn；
#    num_workers=0 时不能开 persistent_workers。
# ------------------------------------------------------------------
import platform  # noqa: E402

_start_method = 'spawn' if platform.system() == 'Windows' else 'fork'
_num_workers = 0 if platform.system() == 'Windows' else 4

train_dataloader['num_workers'] = _num_workers
train_dataloader['persistent_workers'] = _num_workers > 0
val_dataloader['num_workers'] = _num_workers
val_dataloader['persistent_workers'] = _num_workers > 0

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method=_start_method, opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
