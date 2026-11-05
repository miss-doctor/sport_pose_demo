"""关键点与骨架可视化（纯 OpenCV 实现，不依赖可视化后端）。"""

import cv2
import numpy as np

# COCO 17 骨架连线（0 索引）
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # 面部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # 上肢
    (5, 11), (6, 12), (11, 12),                # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),    # 下肢
]

# 每个部位的连线颜色（BGR）
LIMB_COLORS = [
    (255, 200, 100), (255, 200, 100), (255, 200, 100), (255, 200, 100),
    (80, 180, 255), (230, 120, 60), (230, 120, 60), (60, 200, 120),
    (60, 200, 120),
    (80, 180, 255), (80, 180, 255), (80, 180, 255),
    (200, 90, 220), (200, 90, 220), (220, 140, 60), (220, 140, 60),
]

KEYPOINT_COLORS = [
    (60, 60, 230),      # 0 nose - 红
    (200, 200, 230),    # 1 left_eye
    (200, 200, 230),    # 2 right_eye
    (160, 200, 230),    # 3 left_ear
    (160, 200, 230),    # 4 right_ear
    (230, 160, 60),     # 5 left_shoulder - 蓝
    (60, 160, 230),     # 6 right_shoulder - 橙
    (230, 130, 40),     # 7 left_elbow
    (40, 160, 230),     # 8 right_elbow
    (230, 110, 30),     # 9 left_wrist
    (30, 160, 230),     # 10 right_wrist
    (230, 160, 60),     # 11 left_hip
    (60, 160, 230),     # 12 right_hip
    (230, 150, 50),     # 13 left_knee
    (50, 160, 230),     # 14 right_knee
    (230, 140, 40),     # 15 left_ankle
    (40, 160, 230),     # 16 right_ankle
]


def draw_pose(img,
              keypoints,
              scores,
              keypoint_score_thr=0.3,
              limb_width=3,
              keypoint_radius=4,
              bbox=None,
              bbox_color=(80, 200, 120),
              bbox_width=2):
    """在图像上绘制单人骨架。

    Args:
        img (np.ndarray): BGR 图像，会被原地修改。
        keypoints (np.ndarray): (17, 2) 坐标。
        scores (np.ndarray): (17,) 置信度。
        keypoint_score_thr (float): 低于该值的点不绘制。默认 0.3。
        limb_width (int): 骨架线宽。默认 3。
        keypoint_radius (int): 关键点半径。默认 4。
        bbox (np.ndarray | None): (4,) xyxy 检测框。默认 None。
        bbox_color (tuple): 检测框颜色。默认绿色。
        bbox_width (int): 检测框线宽。默认 2。

    Returns:
        np.ndarray: 绘制后的图像。
    """
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, bbox_width)

    visible = scores >= keypoint_score_thr

    for (i, j), color in zip(SKELETON, LIMB_COLORS):
        if not (visible[i] and visible[j]):
            continue
        p1 = tuple(np.round(keypoints[i]).astype(int))
        p2 = tuple(np.round(keypoints[j]).astype(int))
        if not (_in_bounds(img, p1) and _in_bounds(img, p2)):
            continue
        cv2.line(img, p1, p2, color, limb_width, cv2.LINE_AA)

    for idx in range(len(KEYPOINT_COLORS)):
        if not visible[idx]:
            continue
        p = tuple(np.round(keypoints[idx]).astype(int))
        if not _in_bounds(img, p):
            continue
        cv2.circle(img, p, keypoint_radius, KEYPOINT_COLORS[idx], -1,
                   cv2.LINE_AA)
        cv2.circle(img, p, keypoint_radius, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def draw_angle_labels(img,
                      keypoints,
                      scores,
                      angles,
                      min_score=0.3,
                      font_scale=0.5,
                      thickness=1):
    """在关节旁标注角度数值。

    Args:
        img (np.ndarray): BGR 图像，原地修改。
        keypoints (np.ndarray): (17, 2)。
        scores (np.ndarray): (17,)。
        angles (dict): compute_joint_angles 的输出。
        min_score (float): 置信度阈值。默认 0.3。
        font_scale (float): 字号。默认 0.5。
        thickness (int): 字重。默认 1。

    Returns:
        np.ndarray: 绘制后的图像。
    """
    from .analysis.angles import JOINT_DEFS, KEYPOINT_INDEX

    for joint_name, angle in angles.items():
        if angle is None or joint_name not in JOINT_DEFS:
            continue
        vertex_name = JOINT_DEFS[joint_name][0]
        idx = KEYPOINT_INDEX[vertex_name]
        if scores[idx] < min_score:
            continue
        x, y = np.round(keypoints[idx]).astype(int)
        if not _in_bounds(img, (int(x), int(y))):
            continue
        text = f"{angle:.0f}"
        cv2.putText(img, text, (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (40, 40, 40),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
    return img


def _in_bounds(img, point, margin=3):
    h, w = img.shape[:2]
    x, y = point
    return -margin <= x < w + margin and -margin <= y < h + margin
