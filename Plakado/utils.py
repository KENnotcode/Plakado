import cv2
import os

def draw_boxes(img, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        text = det.get('text', '')
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if text:
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return img

def save_output(img, out_dir, filename):
    out_path = os.path.join(out_dir, f"annotated_{filename}")
    cv2.imwrite(out_path, img)
