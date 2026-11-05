"""MMDet(YOLO) + MMPose(RTMPose) 两阶段人体姿态推理流水线。

检测阶段用 MMDetection 中的 YOLO 系列检测器（默认 YOLOX-s，可切
RTMDet），姿态阶段用 MMPose 的 RTMPose-m。
"""

import time

import numpy as np

from .visualize import draw_angle_labels, draw_pose


def _to_numpy(x):
    """兼容 tensor / ndarray 的 numpy 转换。"""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=np.float32)


class SportPosePipeline:
    """两阶段姿态推理。

    姿态模型可以省略（传 None），此时流水线只做人体检测，
    用于生成 RKNN 校准集等只需要检测框的场景。

    Args:
        det_config (str): MMDet 配置文件路径。
        det_checkpoint (str): MMDet 权重路径。
        pose_config (str | None): MMPose 配置文件路径，None 表示不加载。
        pose_checkpoint (str | None): MMPose 权重路径，None 表示不加载。
        device (str): 推理设备，如 'cpu' / 'cuda:0'。默认 'cpu'。
        bbox_thr (float): 人体检测框置信度阈值。默认 0.5。
        person_class (int): COCO 中 person 类别 id。默认 0。
        kpt_thr (float): 关键点置信度阈值。默认 0.3。
    """

    def __init__(self,
                 det_config,
                 det_checkpoint,
                 pose_config=None,
                 pose_checkpoint=None,
                 device="cpu",
                 bbox_thr=0.5,
                 person_class=0,
                 kpt_thr=0.3):
        # 延迟导入：让本模块在未安装 OpenMMLab 时也能被 import（便于文档生成）
        from mmdet.apis import init_detector

        self.det_model = init_detector(det_config, det_checkpoint,
                                       device=device)
        if pose_config and pose_checkpoint:
            from mmpose.apis import init_model
            self.pose_model = init_model(pose_config, pose_checkpoint,
                                         device=device)
        else:
            self.pose_model = None
        self.device = device
        self.bbox_thr = bbox_thr
        self.person_class = person_class
        self.kpt_thr = kpt_thr
        self.det_time = 0.0
        self.pose_time = 0.0

    # ------------------------------------------------------------------
    def detect(self, img):
        """仅执行人体检测。

        Args:
            img (np.ndarray): BGR 图像。

        Returns:
            np.ndarray: (N, 4) xyxy 格式的人体框。没有人时为空数组。
        """
        from mmengine.registry import DefaultScope
        from mmdet.apis import inference_detector

        # MMDet 与 MMPose 共用 TRANSFORMS 注册表，但默认作用域不同。
        # 不切作用域的话，PackDetInputs 会报 "not in the mmpose::transform
        # registry"。用上下文保证检测阶段始终在 mmdet 作用域下构建流水线。
        t0 = time.perf_counter()
        with DefaultScope.overwrite_default_scope("mmdet"):
            result = inference_detector(self.det_model, img)
        self.det_time = time.perf_counter() - t0

        pred = result.pred_instances
        boxes = pred.bboxes.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        keep = (labels == self.person_class) & (scores >= self.bbox_thr)
        return boxes[keep]

    def estimate(self, img, boxes):
        """对给定人体框执行关键点估计。

        Args:
            img (np.ndarray): BGR 图像。
            boxes (np.ndarray): (N, 4) xyxy 人体框。

        Returns:
            list[np.ndarray]: 每个元素为 (17, 2) 关键点坐标。
            list[np.ndarray]: 每个元素为 (17,) 关键点置信度。
        """
        from mmengine.registry import DefaultScope
        from mmpose.apis import inference_topdown

        if boxes is None or len(boxes) == 0:
            return [], []

        t0 = time.perf_counter()
        with DefaultScope.overwrite_default_scope("mmpose"):
            samples = inference_topdown(
                self.pose_model, img, np.asarray(boxes, dtype=np.float32),
                bbox_format="xyxy")
        self.pose_time = time.perf_counter() - t0

        keypoints_list, scores_list = [], []
        for sample in samples:
            inst = sample.pred_instances
            # CPU 上 inference_topdown 直接返回 ndarray，GPU 上返回 tensor，
            # 统一成 ndarray 再处理。
            keypoints_list.append(_to_numpy(inst.keypoints[0]))
            scores_list.append(_to_numpy(inst.keypoint_scores[0]))
        return keypoints_list, scores_list

    # ------------------------------------------------------------------
    def __call__(self, img, with_angles=True):
        """对单帧执行完整两阶段推理。

        Args:
            img (np.ndarray): BGR 图像。
            with_angles (bool): 是否计算关节角度。默认 True。

        Returns:
            list[dict]: 每人一条结果，含 bbox / keypoints / scores / angles。
        """
        boxes = self.detect(img)
        keypoints_list, scores_list = self.estimate(img, boxes)

        results = []
        for bbox, kpts, scores in zip(boxes, keypoints_list, scores_list):
            item = {
                "bbox": bbox.astype(np.float32),
                "keypoints": kpts,
                "scores": scores,
            }
            if with_angles:
                from .analysis import compute_joint_angles
                item["angles"] = compute_joint_angles(kpts, scores,
                                                      self.kpt_thr)
            results.append(item)
        return results

    # ------------------------------------------------------------------
    def visualize(self, img, results, draw_angles=True, **draw_kwargs):
        """把推理结果绘制到图像副本上。

        Args:
            img (np.ndarray): BGR 图像。
            results (list[dict]): __call__ 的返回值。
            draw_angles (bool): 是否标注关节角度。默认 True。

        Returns:
            np.ndarray: 绘制后的图像副本。
        """
        canvas = img.copy()
        for item in results:
            draw_pose(canvas,
                      item["keypoints"],
                      item["scores"],
                      keypoint_score_thr=self.kpt_thr,
                      bbox=item["bbox"],
                      **draw_kwargs)
            if draw_angles and "angles" in item:
                draw_angle_labels(canvas, item["keypoints"], item["scores"],
                                  item["angles"], min_score=self.kpt_thr)
        return canvas

    @property
    def last_infer_time(self):
        """最近一帧的检测 + 姿态总耗时（秒）。"""
        return self.det_time + self.pose_time
