import numpy as np
from typing import Dict, Any, List, Tuple
from .base import Evaluator
from ..core.model_spec import Model_Spec
from ..core.inference_result import InferenceResult

class ObjectDetectionEvaluator(Evaluator):
    """
    객체 탐지 성능 평가 모듈.
    프레임워크 종속적인 NMS(Non-Maximum Suppression)를 배제하고,
    순수 Numpy 배열 알고리즘으로 NMS 로직과 평가지표를 도출.
    """
    def __init__(self, **eval_options):
        # 딕셔너리에서 설정값을 추출 (기본값 설정)
        self.conf_threshold = eval_options.get("conf_threshold", 0.25)
        self.iou_threshold = eval_options.get("iou_threshold", 0.45)
        self.image_size = eval_options.get("image_size", 640)  # 640 고정 방지

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """추론 결과를 받아 헬퍼 메서드로 각 연산을 위임하여 채점함."""
        metrics = {}
        
        # 1. 런타임 결과물(outputs) 추출
        # 보통 YOLO 모델은 [(Batch, Num_Anchors, 85)] 등 1개의 큰 텐서를 도출함
        pred_key = list(result.outputs.keys())[0]
        preds = result.outputs[pred_key]
        
        # 2. 정답지(labels) 추출 및 1D(Batch 전개) 평탄화 처리 (SOLID)
        raw_labels = result.labels
        labels = []
        if isinstance(raw_labels, list):
            for batch_labels in raw_labels:
                if isinstance(batch_labels, list):
                    labels.extend(batch_labels)
                else:
                    labels.append(batch_labels)
        else:
            labels = raw_labels
        
        batch_size = len(labels) if isinstance(labels, (list, tuple, np.ndarray)) else 0
        metrics["Total Samples"] = batch_size

        # 3. 각 지표 계산 역할을 프라이빗 헬퍼 메서드들에게 위임
        box_metrics = self._calculate_detection_metrics(preds, labels)
        metrics.update(box_metrics)

        latency_metrics = self._calculate_latency_metrics(result.timing_records)
        metrics.update(latency_metrics)
            
        return metrics

    def _nms_pure_numpy(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """내부 헬퍼 함수: 파이토치 의존성을 제거한 순수 Numpy NMS 알고리즘."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # 신뢰도 점수가 높은 순으로 정렬된 인덱스
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
                
            # 겹치는 영역(Intersection) 계산
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # IoU 연산
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # IoU가 임계치(Threshold)보다 작은(겹치지 않는) 인덱스들만 남김
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep

    def _process_ground_truths(self, labels: Any) -> List[List[float]]:
        """정답지를 파싱하여 [img_idx, class_id, 1.0, x1, y1, x2, y2] 형태로 평탄화."""
        all_gts = []
        if labels is None:
            return all_gts
            
        for idx in range(len(labels)):
            if len(labels[idx]) == 0:
                continue
            img_gts = np.array(labels[idx])
            cls_ids = img_gts[:, 0]
            # 정규화 좌표 -> 모델 입력 해상도로 동적 복원 (하드코딩 제거)
            size = self.image_size
            cx, cy, w, h = img_gts[:, 1] * size, img_gts[:, 2] * size, img_gts[:, 3] * size, img_gts[:, 4] * size
            for j in range(len(img_gts)):
                all_gts.append([idx, cls_ids[j], 1.0, cx[j]-w[j]/2, cy[j]-h[j]/2, cx[j]+w[j]/2, cy[j]+h[j]/2])
        return all_gts

    def _process_predictions(self, preds: np.ndarray) -> Tuple[List[List[float]], int]:
        """예측 텐서를 파싱하고, Numpy NMS를 적용해 [img_idx, class_id, conf, x1, y1, x2, y2] 형태로 평탄화."""
        all_preds = []
        total_detected = 0
        
        for idx in range(preds.shape[0]):
            img_preds = preds[idx]
            try:
                p_cx, p_cy, p_w, p_h = img_preds[:, 0], img_preds[:, 1], img_preds[:, 2], img_preds[:, 3]
                boxes = np.stack([p_cx - p_w/2, p_cy - p_h/2, p_cx + p_w/2, p_cy + p_h/2], axis=1)
                
                obj_conf = img_preds[:, 4]
                mask = obj_conf > self.conf_threshold
                filtered_boxes = boxes[mask]
                filtered_conf = obj_conf[mask]
                filtered_class_probs = img_preds[mask, 5:]
                
                if len(filtered_boxes) > 0:
                    class_ids = np.argmax(filtered_class_probs, axis=1)
                    final_confs = filtered_conf * np.max(filtered_class_probs, axis=1)
                    
                    keep_indices = self._nms_pure_numpy(filtered_boxes, final_confs, self.iou_threshold)
                    total_detected += len(keep_indices)
                    
                    for ki in keep_indices:
                        all_preds.append([idx, class_ids[ki], final_confs[ki]] + filtered_boxes[ki].tolist())
            except Exception:
                pass
        return all_preds, total_detected

    def _calculate_ap_per_class(self, c_preds: np.ndarray, c_gts: np.ndarray) -> float:
        """단일 클래스의 예측 상자들과 정답 박스 간의 IoU를 측정하여 AP@0.5를 도출합니다."""
        c_preds = c_preds[c_preds[:, 2].argsort()[::-1]]
        nd, nt = len(c_preds), len(c_gts)
        tp, fp = np.zeros(nd), np.zeros(nd)
        
        # 이미지별 매칭 식별용 딕셔너리 할당
        gt_matched = {i: np.zeros(len(c_gts[c_gts[:, 0] == i])) for i in np.unique(c_gts[:, 0])}
        
        for d_idx in range(nd):
            img_idx = c_preds[d_idx, 0]
            pred_box = c_preds[d_idx, 3:7]
            
            img_gts = c_gts[c_gts[:, 0] == img_idx]
            if len(img_gts) == 0:
                fp[d_idx] = 1
                continue
                
            # 순수 Numpy 면적 및 IoU 연산
            ixmin = np.maximum(img_gts[:, 3], pred_box[0])
            iymin = np.maximum(img_gts[:, 4], pred_box[1])
            ixmax = np.minimum(img_gts[:, 5], pred_box[2])
            iymax = np.minimum(img_gts[:, 6], pred_box[3])
            iw, ih = np.maximum(ixmax - ixmin, 0.), np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            
            area_gt = (img_gts[:, 5] - img_gts[:, 3]) * (img_gts[:, 6] - img_gts[:, 4])
            area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            ious = inters / np.maximum(area_gt + area_pred - inters, 1e-6)
            
            jmax = np.argmax(ious)
            if np.max(ious) >= 0.5 and not gt_matched[img_idx][jmax]:
                tp[d_idx] = 1
                gt_matched[img_idx][jmax] = 1
            else:
                fp[d_idx] = 1
                
        fpc, tpc = np.cumsum(fp), np.cumsum(tp)
        recalls = tpc / np.maximum(nt, 1e-6)
        precisions = tpc / np.maximum(fpc + tpc, 1e-6)
        
        # PR(Precision-Recall) 곡선 보정
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            
        # 적분 수행 (AP 연산)
        idx_res = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[idx_res + 1] - recalls[idx_res]) * precisions[idx_res + 1])
        return float(ap)

    def _calculate_mean_ap(self, all_preds: List[List[float]], all_gts: List[List[float]]) -> float:
        """모든 예측/정답값을 클래스별로 쪼갠 뒤 AP를 구하고 전체 평균(mAP)을 산출함."""
        if len(all_preds) == 0 or len(all_gts) == 0:
            return 0.0
            
        pred_arr = np.array(all_preds)
        gt_arr = np.array(all_gts)
        unique_classes = np.unique(gt_arr[:, 1])
        ap_list = []
        
        for c in unique_classes:
            c_preds = pred_arr[pred_arr[:, 1] == c]
            c_gts = gt_arr[gt_arr[:, 1] == c]
            if len(c_preds) == 0:
                ap_list.append(0.0)
                continue
            if len(c_gts) == 0:
                continue
            
            ap = self._calculate_ap_per_class(c_preds, c_gts)
            ap_list.append(ap)
            
        return float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0

    def _calculate_detection_metrics(self, preds: np.ndarray, labels: Any) -> Dict[str, float]:
        """분리된 여러 프라이빗 헬퍼 함수에 채점을 위임하여 벤치마크 지표를 반환."""
        # 1. 정답지 및 예측값 파싱 (NMS 적용 포함)
        all_gts = self._process_ground_truths(labels)
        all_preds, total_detected = self._process_predictions(preds)
        
        # 2. mAP 연산 엔진 구동
        mean_ap = self._calculate_mean_ap(all_preds, all_gts)

        return {
            "mAP@0.5": mean_ap,
            "Average Detections": float(total_detected / max(1, preds.shape[0]))
        }

    def _calculate_latency_metrics(self, timing_records: List[float]) -> Dict[str, float]:
        """내부 헬퍼 함수: 실행 소요 시간(Latency) 및 처리량(FPS)을 기반으로 지표를 계산함."""
        if not timing_records:
            return {}
            
        avg_latency = float(np.mean(timing_records))
        p99_latency = float(np.percentile(timing_records, 99))
        
        # Latency 값이 밀리초(ms) 단위이므로, 1초(1000ms)를 지연 시간으로 나누면 1초당 프레임 수(FPS) 도출
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        
        return {
            "Average Latency (ms)": avg_latency,
            "P99 Latency (ms)": p99_latency,
            "FPS": fps
        }

    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        """이 평가기가 YOLO 같은 객체 탐지(Object Detection) 모델을 채점할 수 있는지 검사함."""
        task_name = str(getattr(model_spec, "task", ""))
        return "OBJECT_DETECTION" in task_name

    def get_metric_names(self) -> List[str]:
        """해당 모듈에서 반환 가능한 지표 이름의 목록을 반환함."""
        return [
            "mAP@0.5", "Average Detections", 
            "Average Latency (ms)", "P99 Latency (ms)", "FPS", "Total Samples"
        ]
