import os
import sys
import argparse
import logging
import time
from pathlib import Path
import glob
import json

import numpy as np
from tqdm import tqdm
import cv2
import yaml
import psutil  # Added for CPU, memory, and disk information

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from edgetpumodel import EdgeTPUModel
from utils import resize_and_pad, get_image_tensor, save_one_json, coco80_to_coco91_class

# Function to get CPU temperatures for each core (Linux only)
def get_cpu_temperatures():
    try:
        # Retrieve CPU temperature for each core (Linux only)
        temperatures = psutil.sensors_temperatures()
        
        if 'coretemp' in temperatures:
            core_temps = temperatures['coretemp']
            core_temps_dict = {f"Core {i}": temp.current for i, temp in enumerate(core_temps)}
            return core_temps_dict
        else:
            logger.warning("Could not retrieve core temperatures.")
            return None
    except Exception as e:
        logger.warning(f"Error getting CPU temperatures: {e}")
        return None

# Function to get Edge TPU temperature
def get_edge_tpu_temperature():
    try:
        # Replace 'apex_0' with the appropriate sysfs node for your Edge TPU
        temperature_file = '/sys/class/apex/apex_0/temp'
        if os.path.exists(temperature_file):
            with open(temperature_file, 'r') as f:
                temperature = int(f.read().strip())
                return temperature / 1000.0  # Convert to degrees Celsius
        else:
            logger.warning("Could not retrieve Edge TPU temperature.")
            return None
    except Exception as e:
        logger.warning(f"Error getting Edge TPU temperature: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser("EdgeTPU test runner")
    parser.add_argument("--model", "-m", help="weights file", required=True)
    parser.add_argument("--bench_speed", action='store_true', help="run speed test on dummy data")
    parser.add_argument("--bench_image", action='store_true', help="run detection test")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="model confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--names", type=str, default='data/coco.yaml', help="Names file")
    parser.add_argument("--image", "-i", type=str, help="Image file to run detection on")
    parser.add_argument("--device", type=int, default=0, help="Image capture device to run live detection")
    parser.add_argument("--stream", action='store_true', help="Process a stream")
    parser.add_argument("--bench_coco", action='store_true', help="Process a stream")
    parser.add_argument("--coco_path", type=str, help="Path to COCO 2017 Val folder")
    parser.add_argument("--quiet", "-q", action='store_true', help="Disable logging (except errors)")
    parser.add_argument("--video", type=str, help="Video file to run detection on")  # New argument

    args = parser.parse_args()

    if args.quiet:
        logging.disable(logging.CRITICAL)
        logger.disabled = True

    if args.stream and args.image:
        logger.error("Please select either an input image, a stream, or a video")
        exit(1)

    model = EdgeTPUModel(args.model, args.names, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)
    input_size = model.get_image_size()

    x = (255 * np.random.random((3, *input_size))).astype(np.uint8)
    model.forward(x)

    conf_thresh = 0.25
    iou_thresh = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000

    if args.bench_speed:
        logger.info("Performing test run")
        n_runs = 100

        inference_times = []
        nms_times = []
        total_times = []

        for i in tqdm(range(n_runs)):
            x = (255 * np.random.random((3, *input_size))).astype(np.float32)

            pred = model.forward(x)
            tinference, tnms = model.get_last_inference_time()

            inference_times.append(tinference)
            nms_times.append(tnms)
            total_times.append(tinference + tnms)

        inference_times = np.array(inference_times)
        nms_times = np.array(nms_times)
        total_times = np.array(total_times)

        logger.info("Inference time (EdgeTPU): {:1.2f} +- {:1.2f} ms".format(inference_times.mean() / 1e-3,
                                                                               inference_times.std() / 1e-3))
        logger.info("NMS time (CPU): {:1.2f} +- {:1.2f} ms".format(nms_times.mean() / 1e-3, nms_times.std() / 1e-3))
        fps = 1.0 / total_times.mean()
        logger.info("Mean FPS: {:1.2f}".format(fps))

    elif args.bench_image:
        logger.info("Testing on Zidane image")
        model.predict("./data/images/zidane.jpg")

    elif args.bench_coco:
        logger.info("Testing on COCO dataset")

        model.conf_thresh = 0.001
        model.iou_thresh = 0.65

        coco_glob = os.path.join(args.coco_path, "*.jpg")
        images = glob.glob(coco_glob)

        logger.info("Looking for: {}".format(coco_glob))
        ids = [int(os.path.basename(i).split('.')[0]) for i in images]

        out_path = "./coco_eval"
        os.makedirs("./coco_eval", exist_ok=True)

        logger.info("Found {} images".format(len(images)))

        class_map = coco80_to_coco91_class()

        predictions = []

        for image in tqdm(images):
            res = model.predict(image, save_img=False, save_txt=False)
            save_one_json(res, predictions, Path(image), class_map)

        pred_json = os.path.join(out_path,
                                 "{}_predictions.json".format(os.path.basename(args.model)))

        with open(pred_json, 'w') as f:
            json.dump(predictions, f, indent=1)

    elif args.image is not None:
        logger.info("Testing on user image: {}".format(args.image))
        model.predict(args.image)

    elif args.stream:
        logger.info("Opening stream on device: {}".format(args.device))

        cam = cv2.VideoCapture(args.device)

        while True:
            try:
                res, image = cam.read()

                if res is False:
                    logger.error("Empty image received")
                    break
                else:
                    full_image, net_image, pad = get_image_tensor(image, input_size[0])
                    pred = model.forward(net_image)
                    model.process_predictions(pred[0], full_image, pad)

                    tinference, tnms = model.get_last_inference_time()
                    logger.info("Frame done in {}".format(tinference + tnms))
            except KeyboardInterrupt:
                break

        cam.release()

    elif args.video:
        logger.info("Testing on user video: {}".format(args.video))

        # Open the video file
        cap = cv2.VideoCapture(args.video)

        total_time_sum = 0  # Initialize total_time_sum
        frame_count = 0

        while True:
            try:
                res, frame = cap.read()

                if not res:
                    logger.info("End of video")
                    break

                # Perform inference on the current frame
                full_image, net_image, pad = get_image_tensor(frame, input_size[0])
                pred = model.forward(net_image)
                model.process_predictions(pred[0], full_image, pad)

                tinference, tnms = model.get_last_inference_time()

                total_time = tinference + tnms
                total_time_sum += total_time  # Accumulate total_time_sum
                frame_count += 1

                # Display CPU temperatures, Edge TPU temperature, memory usage, and disk space every 10 frames
                if frame_count % 10 == 0:
                    # Get CPU temperatures for each core
                    cpu_temperatures = get_cpu_temperatures()
                    if cpu_temperatures:
                        for core, temp in cpu_temperatures.items():
                            logger.info(f"{core} Temperature: {temp}°C")
                    else:
                        logger.warning("Unable to retrieve CPU temperatures.")

                    # Get Edge TPU temperature
                    edge_tpu_temp = get_edge_tpu_temperature()
                    if edge_tpu_temp is not None:
                        logger.info(f"Edge TPU Temperature: {edge_tpu_temp}°C")
                    else:
                        logger.warning("Unable to retrieve Edge TPU temperature.")

                    # Get memory usage
                    memory_info = psutil.virtual_memory()
                    used_memory = memory_info.used / (1024 ** 2)
                    total_memory = memory_info.total / (1024 ** 2)
                    logger.info(f"Memory Usage: {used_memory:.2f} MB used / {total_memory:.2f} MB total")

                    # Get disk I/O statistics
                    disk_io = psutil.disk_io_counters()
                    logger.info(f"Disk I/O: Read Count: {disk_io.read_count}, Write Count: {disk_io.write_count}, "
                                f"Read Bytes: {disk_io.read_bytes}, Write Bytes: {disk_io.write_bytes}")

                    fps = frame_count / total_time_sum
                    logger.info("FPS: ({})".format(fps))

                logger.info("Frame done in {} (Inference: {} ms, NMS: {} ms)".format(total_time, tinference, tnms))

                # Display the processed frame (optional)
                cv2.imshow('Frame', full_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except KeyboardInterrupt:
                break

        cap.release()
        cv2.destroyAllWindows()

        # Return total_time_sum after processing all frames
        logger.info("Total time for the video: {} ms".format(total_time_sum))
        average_fps = frame_count / total_time_sum
        logger.info("Average FPS: ({})".format(average_fps))

