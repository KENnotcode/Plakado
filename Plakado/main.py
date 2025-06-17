from plakado.detector import PlateDetector
from plakado.ocr_reader import OCRReader
from plakado.utils import draw_boxes, save_output

import cv2
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plakado - Automatic Plate Recognition")
    parser.add_argument("--input", required=True, help="Path to input image or video")
    parser.add_argument("--output", default="output", help="Directory to save output")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLOv8")
    return parser.parse_args()

def process_image(img_path, detector, reader, output_dir):
    img = cv2.imread(img_path)
    detections = detector.detect(img)
    for det in detections:
        cropped = det['crop']
        text = reader.read(cropped)
        det['text'] = text
    img = draw_boxes(img, detections)
    save_output(img, output_dir, os.path.basename(img_path))

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    detector = PlateDetector(conf_thres=args.conf)
    reader = OCRReader()

    if args.input.endswith(('.jpg', '.jpeg', '.png')):
        process_image(args.input, detector, reader, args.output)
    elif args.input.endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(args.input)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(args.output, "annotated_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)
            for det in detections:
                det['text'] = reader.read(det['crop'])
            annotated = draw_boxes(frame, detections)
            out.write(annotated)

        cap.release()
        out.release()
    else:
        print("Unsupported file format")

if __name__ == "__main__":
    main()