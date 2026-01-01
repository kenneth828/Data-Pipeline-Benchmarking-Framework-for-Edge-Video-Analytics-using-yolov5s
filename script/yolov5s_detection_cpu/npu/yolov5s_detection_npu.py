import cv2
import numpy as np
import time
import argparse
import os
import psutil
from pathlib import Path
import tflite_runtime.interpreter as tflite

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

def process_output(output, frame_width, frame_height, conf_threshold, nms_threshold):
    boxes, confidences, class_ids = [], [], []
    for detection in output[0]:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, confidences, class_ids, indices

def main(weights, source, conf_thres, iou_thres, save_csv, info_interval):
    try:
        delegate = tflite.load_delegate('/usr/lib/libvx_delegate.so')
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
        with open('coco.txt', 'r') as f:
            labels = f.read().strip().split('\n')
    except FileNotFoundError:
        labels = None
        print("Warning: coco.txt not found. Object labels will not be displayed.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

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
        input_frame = np.expand_dims(input_frame, axis=0)
        input_frame = (input_frame / 255.0).astype(np.float32)
        input_frame = (input_frame * 255).astype(np.uint8)

        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_frame)
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000

        output = interpreter.get_tensor(output_details[0]['index'])
        boxes, confidences, class_ids, indices = process_output(output, frame_width, frame_height, conf_thres, iou_thres)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                label = labels[class_id] if labels else str(class_id)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if save_csv:
                    csv_file.write(f"{frame_count},{label},{confidence:.2f}\n")

        display_start_time = time.time()
        cv2.imshow('Detection', frame)
        out.write(frame)
        display_time = (time.time() - display_start_time) * 1000

        total_time = inference_time + display_time
        fps = 1000.0 / total_time
        frame_times.append(fps)

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

            avg_fps = sum(frame_times) / len(frame_times)
            print(f"Average FPS: {avg_fps:.2f}")
            frame_times = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    out.release()
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
