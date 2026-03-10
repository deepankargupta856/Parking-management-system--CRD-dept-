import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import deque

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def point_in_poly(point, poly):
    return cv2.pointPolygonTest(poly, point, False) >= 0

def get_polygon_from_box(box):
    x1, y1, x2, y2 = box
    return np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ], dtype=np.int32)

def main():
    video_path = "resources/nie parking video.mp4"
    model = YOLO("yolov8s.pt")
    # Using yolov8m as a pseudo-ground-truth generator to calculate accuracy metrics
    model_gt = YOLO("yolov8m.pt")  
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
        
    ret, frame = cap.read()
    if not ret:
        return
        
    frame = cv2.resize(frame, (1024, 768))
    
    results = model(frame, classes=[2, 7], conf=0.5, verbose=False)[0]
    slots = []
    for box in results.boxes:
        slots.append(get_polygon_from_box(box.xyxy[0].cpu().numpy()))
        
    print(f"Generated {len(slots)} automatic parking slots for tracking...")
    
    SMOOTH_WINDOW = 5
    slot_history = [deque(maxlen=SMOOTH_WINDOW) for _ in slots]
    smoothed_states = [True] * len(slots)
    state_flips = 0
    total_time_eval = 0
    
    latencies = []
    
    gt_all = []
    pred_all = []
    
    frame_count = 0
    MAX_FRAMES = 200  
    
    print(f"Evaluating metrics over {MAX_FRAMES} frames...")
    
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (1024, 768))
        
        # Ground Truth Predictions (Using heavier model)
        res_gt = model_gt(frame, classes=[2, 7], conf=0.4, verbose=False)[0]
        gt_all.append([box.xyxy[0].cpu().numpy() for box in res_gt.boxes])
        
        # Target Model Predictions (with latency tracking)
        start_t = time.time()
        res = model(frame, classes=[2, 7], conf=0.6, verbose=False)[0]
        end_t = time.time()
        
        pred_all.append([{"box": box.xyxy[0].cpu().numpy(), "conf": box.conf[0].cpu().item()} for box in res.boxes])
        
        latency = (end_t - start_t) * 1000
        latencies.append(latency)
        total_time_eval += (end_t - start_t)
        
        # State Flip Tracking
        slot_status = [False] * len(slots)
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            for i, poly in enumerate(slots):
                if point_in_poly((cx, cy), poly):
                    slot_status[i] = True
                    break
        
        for i, status in enumerate(slot_status):
            slot_history[i].append(status)
            is_occupied = sum(slot_history[i]) >= (len(slot_history[i]) // 2 + 1)
            
            if is_occupied != smoothed_states[i]:
                if len(slot_history[i]) == SMOOTH_WINDOW:
                    state_flips += 1
                smoothed_states[i] = is_occupied
        
        frame_count += 1
        if frame_count % 20 == 0:
            print(f"Processed {frame_count}/{MAX_FRAMES} frames...")

    # Metrics Calculation
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    flip_rate = state_flips / total_time_eval if total_time_eval > 0 else 0
    
    # Precision and Recall
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for gt, pred in zip(gt_all, pred_all):
        matched_gt = set()
        for p in pred:
            best_iou = 0
            best_idx = -1
            for j, g in enumerate(gt):
                if j in matched_gt:
                    continue
                iou = calculate_iou(p['box'], g)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= 0.5:
                true_positives += 1
                matched_gt.add(best_idx)
            else:
                false_positives += 1
        false_negatives += len(gt) - len(matched_gt)
        
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # AP approximation
    preds_flat = []
    total_gt = 0
    
    for gt, pred in zip(gt_all, pred_all):
        total_gt += len(gt)
        matched_gt = set()
        pred_sorted = sorted(pred, key=lambda x: x['conf'], reverse=True)
        
        for p in pred_sorted:
            best_iou = 0
            best_idx = -1
            for j, g in enumerate(gt):
                if j in matched_gt:
                    continue
                iou = calculate_iou(p['box'], g)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= 0.5:
                preds_flat.append((p['conf'], 1))
                matched_gt.add(best_idx)
            else:
                preds_flat.append((p['conf'], 0))
                
    preds_flat.sort(key=lambda x: x[0], reverse=True)
    
    tps = np.cumsum([x[1] for x in preds_flat])
    fps = np.cumsum([1 - x[1] for x in preds_flat])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = tps / total_gt if total_gt > 0 else np.zeros_like(tps)
        precisions = tps / (tps + fps)
    
    recalls = np.nan_to_num(recalls)
    precisions = np.nan_to_num(precisions)
    
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    map_50 = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    output = []
    output.append(f"Metric\tEstimated Value")
    output.append(f"Mean Latency\t~{mean_latency:.0f} ms")
    output.append(f"95th Percentile Latency\t~{p95_latency:.0f} ms")
    output.append(f"State Flip Rate (window=5)\t~{flip_rate:.2f} flips/sec")
    output.append("")
    output.append("=== Accuracy Metrics (vs YOLOv8m Pseudo-GT) ===")
    output.append(f"Precision:\t{precision:.4f}")
    output.append(f"Recall:\t{recall:.4f}")
    output.append(f"mAP@0.5:\t{map_50:.4f}")
    
    out_str = "\n".join(output)
    print("\n" + out_str)
    
    with open("metrics_output.txt", "w") as f:
        f.write(out_str)

if __name__ == "__main__":
    main()
