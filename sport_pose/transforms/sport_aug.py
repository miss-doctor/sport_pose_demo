"""体育运动场景数据增强。

设计约束（这是本文件与常见「随手写的增强」最大的区别）
------------------------------------------------------------------
本增强 **只做改变像素值、不改变空间位置的变换**，因此不会让关键点标签
与图像内容错位。

几何类增强（水平翻转、旋转、缩放、裁剪）会移动人体在图像中的位置，
必须同步更新 keypoints / bbox，否则标签就是错的。
这一步交给 MMPose 内置的 `RandomFlip` / `RandomHalfBody` /
`RandomBBoxTransform` 完成，它们会按 `flip_indices` 正确交换左右关键点。

本文件负责运动场景特有的「画质退化」类增强：
1. 运动模糊（高斯 + 方向性线性）
2. 亮度 / 对比度扰动（室外体育光照变化）
3. 高斯噪声（高 ISO 抽帧噪点）
4. JPEG 压缩伪影（低码率视频抽帧）
"""

import random

import cv2
import numpy as np

from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS

__all__ = ["SportMotionAug"]


def _make_motion_kernel(size, angle):
    """构造方向性运动模糊核。

    Args:
        size (int): 核边长（奇数）。
        angle (float): 运动方向角度（弧度）。

    Returns:
        np.ndarray: (size, size) 且和为 1 的模糊核。
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    dx, dy = np.cos(angle), np.sin(angle)
    for i in range(size):
        offset = i - (size - 1) / 2.0
        x = int(round(center + dx * offset))
        y = int(round(center + dy * offset))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    total = kernel.sum()
    if total <= 0:  # 极小概率下退化，回退为均值核
        return np.ones((size, size), dtype=np.float32) / (size * size)
    return kernel / total


@TRANSFORMS.register_module()
class SportMotionAug(BaseTransform):
    """运动场景光度增强。

    Args:
        motion_blur_prob (float): 触发运动模糊的概率。默认 0.5。
        motion_blur_kernel_choices (tuple): 模糊核边长候选（奇数）。
            默认 (3, 5, 7, 9)。
        directional_prob (float): 在运动模糊中使用方向性核（而非各向同性
            均值核）的概率。默认 0.6。
        brightness_range (tuple): 亮度缩放系数范围。默认 (0.7, 1.3)。
        contrast_range (tuple): 对比度缩放系数范围。默认 (0.75, 1.25)。
        brightness_prob (float): 触发亮度/对比度扰动的概率。默认 0.6。
        noise_std_range (tuple): 高斯噪声标准差范围（0-255 灰度）。
            默认 (0, 10)。
        noise_prob (float): 触发高斯噪声的概率。默认 0.3。
        jpeg_quality_range (tuple): JPEG 压缩质量范围。默认 (60, 95)。
        jpeg_prob (float): 触发 JPEG 压缩伪影的概率。默认 0.3。
    """

    def __init__(self,
                 motion_blur_prob=0.5,
                 motion_blur_kernel_choices=(3, 5, 7, 9),
                 directional_prob=0.6,
                 brightness_range=(0.7, 1.3),
                 contrast_range=(0.75, 1.25),
                 brightness_prob=0.6,
                 noise_std_range=(0, 10),
                 noise_prob=0.3,
                 jpeg_quality_range=(60, 95),
                 jpeg_prob=0.3):
        super().__init__()
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_kernel_choices = motion_blur_kernel_choices
        self.directional_prob = directional_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.brightness_prob = brightness_prob
        self.noise_std_range = noise_std_range
        self.noise_prob = noise_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.jpeg_prob = jpeg_prob

    # ------------------------------------------------------------------
    def _motion_blur(self, img):
        ksize = random.choice(self.motion_blur_kernel_choices)
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        if random.random() < self.directional_prob:
            angle = random.uniform(0, np.pi)
            kernel = _make_motion_kernel(ksize, angle)
        else:
            kernel = np.ones((ksize, ksize), dtype=np.float32)
            kernel /= kernel.sum()
        return cv2.filter2D(img, -1, kernel)

    def _brightness_contrast(self, img):
        alpha = random.uniform(*self.contrast_range)
        beta = (random.uniform(*self.brightness_range) - 1.0) * 50.0
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def _gaussian_noise(self, img):
        std = random.uniform(*self.noise_std_range)
        if std <= 0:
            return img
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(img.dtype)

    def _jpeg(self, img):
        quality = int(random.uniform(*self.jpeg_quality_range))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                             quality])
        if not ok:
            return img
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # ------------------------------------------------------------------
    def transform(self, results):
        """执行增强。图像为 BGR 顺序的 uint8 数组（H, W, 3）。

        Args:
            results (dict): MMPose 的 data dict。

        Returns:
            dict: 更新后的 results。
        """
        img = results["img"]
        # 通道数不为 3 时直接跳过（灰度图无 HSV/JPEG 意义）
        if img.ndim != 3 or img.shape[2] != 3:
            return results

        if random.random() < self.motion_blur_prob:
            img = self._motion_blur(img)
        if random.random() < self.brightness_prob:
            img = self._brightness_contrast(img)
        if random.random() < self.noise_prob:
            img = self._gaussian_noise(img)
        if random.random() < self.jpeg_prob:
            img = self._jpeg(img)

        results["img"] = img
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"motion_blur_prob={self.motion_blur_prob}, "
                f"brightness_prob={self.brightness_prob}, "
                f"noise_prob={self.noise_prob}, "
                f"jpeg_prob={self.jpeg_prob})")
