import time
import argparse
import os
import psutil
from pathlib import Path
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from collections import Counter
import yaml

def get_cpu_temperatures():
    try:
        temperatures = {}
        thermal_dir = "/sys/class/thermal/"
        if os.path.exists(thermal_dir):
            for device in os.listdir(thermal_dir):
                if device.startswith("thermal_zone"):
                    with open(os.path.join(thermal_dir, device, "temp")) as f:
                        temperature = int(f.read().strip()) / 1000.0
                        temperatures[device] = temperature
            return temperatures
        else:
            print("Warning: Could not retrieve CPU temperatures.")
            return None
    except Exception as e:
        print(f"Error getting CPU temperatures: {e}")
        return None

def get_memory_usage():
    try:
        memory_info = psutil.virtual_memory()
        used_memory = memory_info.used / (1024 ** 2)
        total_memory = memory_info.total / (1024 ** 2)
        return used_memory, total_memory
    except Exception as e:
        print(f"Error getting memory usage: {e}")
        return None, None

def get_disk_io():
    try:
        disk_io = psutil.disk_io_counters()
        return disk_io.read_count, disk_io.write_count, disk_io.read_bytes, disk_io.write_bytes
    except Exception as e:
        print(f"Error getting disk I/O statistics: {e}")
        return None, None, None, None

def process_output(output, conf_threshold, img_shape):
    boxes, confidences, class_ids = [], [], []
    h, w = img_shape[:2]
    for detection in output:
        if detection[4] > conf_threshold:  # Check if confidence is above threshold
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                box = detection[:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def run_nms(boxes, confidences, class_ids, conf_threshold, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    nms_boxes, nms_confidences, nms_class_ids = [], [], []
    if len(indices) > 0:
        for i in indices.flatten():
            nms_boxes.append(boxes[i])
            nms_confidences.append(confidences[i])
            nms_class_ids.append(class_ids[i])
    return nms_boxes, nms_confidences, nms_class_ids

def main(weights, source, conf_thres, iou_thres, save_csv, info_interval):
    try:
        delegate = tflite.load_delegate('/usr/lib/libethosu_delegate.so')
    except Exception as e:
        print(f"Error loading delegate: {e}")
        return

    interpreter = tflite.Interpreter(
        model_path=weights,
        experimental_delegates=[delegate]
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    try:
        with open('data/coco128.yaml', 'r') as f:
            data = yaml.safe_load(f)
            labels = data['names']
    except FileNotFoundError:
        labels = None
        print("Warning: data/coco128.yaml not found. Object labels will not be displayed.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    frame_times = []

    if save_csv:
        csv_file = open('predictions.csv', 'a')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from video.")
            break

        frame_height, frame_width = frame.shape[:2]
        input_frame = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)

        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_frame)
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # convert to milliseconds

        output = interpreter.get_tensor(output_details[0]['index'])[0]

        boxes, confidences, class_ids = process_output(output, conf_thres, frame.shape)
        boxes, confidences, class_ids = run_nms(boxes, confidences, class_ids, conf_thres, iou_thres)

        detection_summary = Counter()
        for class_id in class_ids:
            label = labels[class_id] if labels else str(class_id)
            detection_summary[label] += 1

        if save_csv:
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                label = labels[class_ids[i]] if labels else str(class_ids[i])
                confidence = confidences[i]
                csv_file.write(f"{frame_count},{label},{confidence:.2f},{x},{y},{w},{h}\n")

        if frame_count > 0:  # Skip the first frame for FPS calculation
            frame_times.append(inference_time)

        # Calculate the FPS for the current frame
        if frame_count > 1:
            avg_fps = 1000.0 / np.mean(frame_times)
        else:
            avg_fps = 1000.0 / inference_time

        summary_str = ', '.join([f"{label}: {count}" for label, count in detection_summary.items()])
        print(f"Frame {frame_count}: Inference={inference_time:.2f}ms, Detections: {summary_str}")

        if frame_count % info_interval == 0:
            cpu_temps = get_cpu_temperatures()
            if cpu_temps:
                for core, temp in cpu_temps.items():
                    print(f"Core {core}: {temp}°C")

            used_memory, total_memory = get_memory_usage()
            if used_memory is not None and total_memory is not None:
                print(f"Memory Usage: Used={used_memory:.2f} MB, Total={total_memory:.2f} MB")

            read_count, write_count, read_bytes, write_bytes = get_disk_io()
            if read_count is not None and write_count is not None and read_bytes is not None and write_bytes is not None:
                print(f"Disk I/O Statistics: Read Count={read_count}, Write Count={write_count}, Read Bytes={read_bytes}, Write Bytes={write_bytes}")

            print(f"Average FPS: {avg_fps:.2f}")

        frame_count += 1

    cap.release()
    if save_csv:
        csv_file.close()
    cv2.destroyAllWindows()
    print("Video processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s-int8-224.tflite', help='model path')
    parser.add_argument('--source', type=str, default='video_test.mp4', help='video source path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--info-interval', type=int, default=10, help='interval to display CPU, memory, and disk information (in frames)')
    args = parser.parse_args()

    main(args.weights, args.source, args.conf_thres, args.iou_thres, args.save_csv, args.info_interval)
