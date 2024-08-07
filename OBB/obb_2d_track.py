from ultralytics import YOLO
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

# Define utility functions
def draw_bounding_polygon_with_label(frame, box, label, track_id, color):
    """
    Draw bounding polygon and class label with enhancements.
    """
    # Draw bounding polygon
    cv2.polylines(frame, [np.array(box, np.int32).reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)

    # Prepare text
    label_text = f"{label} ID:{track_id}"
    # Calculate width and height of the text box
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

    # Draw a filled rectangle to put the class label
    cv2.rectangle(frame, (box[0][0], box[0][1] - 30), (box[0][0] + text_width, box[0][1]), color, cv2.FILLED)
    # Draw class label text
    cv2.putText(frame, label_text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def convert_xyxy_to_xyxyxyxy(coords):
    """
    Convert a bounding box from xyxy format to xyxyxyxy format.
    """
    return [(coords[0], coords[1]), (coords[2], coords[1]), (coords[2],coords[3]), (coords[0], coords[3])]


def calculate_iou_obb(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) for two oriented bounding boxes (OBBs).
    Each box is represented in xyxyxyxy format.
    """
    # Convert boxes to polygons
    polyA = Polygon(boxA)
    polyB = Polygon(boxB)

    # Calculate intersection and union
    inter_area = polyA.intersection(polyB).area
    union_area = polyA.area + polyB.area - inter_area

    # Calculate IoU
    iou = inter_area / union_area

    return iou


# Load a model
model_2d = YOLO("yolov9e.pt")  # load a built-in model
model_obb = YOLO("runs/obb/train8/weights/best.pt")  # Your custom model
# Predict with the model
video = cv2.VideoCapture("1816_cam3_conflict_bev.mp4")
# Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Initialize VideoWriter object
frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('results.mp4', fourcc, fps, (frame_width, frame_height))

labels_map_coco = {
    0: "pedestrian",
    #1: "bicycle",
    1: "cyclist", # changed to cyclist
    2: "car",
    3: "motorcyclist",
    5: "bus",
    #6: "train",
    7: "truck",
    #9: "traffic light",
    #10: "fire hydrant",
    #11: "stop sign",
    #12: "parking meter",
    #13: "bench"
}

class_mapping = {
    0:"car",
    1:"truck",
    2:"van",
    3:"bus",
    4:"pedestrian",
    5:"cyclist",
    6:"tricyclist",
    7:"motorcyclist"
}

features = {'id': [],
            'class': [],
            'xywhr': [],
            'corners_obb': [],
            'corners_2d': [],
            'timestamp': []
        }

colors = {class_id: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for class_id in labels_map_coco.keys()}

for i in tqdm(range(frames), desc='Processing frames'):
    #print(f'index {i}')
    ret, frame = video.read()

    if not ret:
        continue
    results_obb = model_obb.track(frame, persist=True , conf=0.3, iou=0.5, tracker = "botsort.yaml", save = False)  # predict on an image
    results_2d = model_2d.track(frame, persist=True , conf=0.3, iou=0.5, tracker = "botsort.yaml", save = False)  # predict on an image
    # Get result values
    boxes_obb = results_obb[0].obb
    boxes_2d = results_2d[0].boxes
    #labels_2d = [labels_map.get(int(det.cls), "misc") for det in boxes_2d]
    iou_threshold = 0.3  # Define an IoU threshold
    try:
        track_ids = results_2d[0].boxes.id.int().cpu().tolist()
    except Exception as e:
        track_ids = []
    for det_2d, track_id in zip(boxes_2d, track_ids):
        # Get the bounding box in xyxy format
        box_2d = convert_xyxy_to_xyxyxyxy([int(x) for x in det_2d.xyxy[0]])
        # Find the matching OBB
        best_iou = 0
        for det_obb in boxes_obb:
            # Get the bounding box in xyxyxyxy format
            corners = det_obb.xyxyxyxy.cpu().numpy().tolist()[0]
            box_obb = [(int(corner[0]), int(corner[1])) for corner in corners]
            # Calculate IoU
            iou = calculate_iou_obb(box_2d, box_obb)
            if iou > best_iou:
                best_iou = iou
                best_box_obb = det_obb
        flag_obb = False
        if best_iou > iou_threshold:
            corners = best_box_obb.xyxyxyxy.cpu().numpy().tolist()[0]
            box = [(int(corner[0]), int(corner[1])) for corner in corners]
            label = class_mapping.get(int(best_box_obb.cls),"misc")
            xywhr = best_box_obb.xywhr.cpu().numpy().tolist()[0]
            flag_obb = True
        else:
            box = box_2d
            label = labels_map_coco.get(int(det_2d.cls), "misc")
            xywhr = np.nan
        # Draw bounding box
        # Get color for the class
        color = colors.get(int(det_2d.cls), (0, 0, 0))
        draw_bounding_polygon_with_label(frame, box, label, track_id, color)
        features['xywhr'] += [xywhr]
        if flag_obb:
            features['corners_obb'] += [box]
        else:
            features['corners_obb'] += [np.nan]
        features['corners_2d'] += [box_2d]
        features['class'] += [label]
        features['id'] += [track_id]
        features['timestamp'] += [video.get(cv2.CAP_PROP_POS_MSEC)]
    # Write the processed frame to the video file
    out.write(frame)
pd.DataFrame(features).to_csv('1816_cam3_featObb.csv', index=False)
#pd.DataFrame(features2d).to_csv(f'features2d.csv', index=False)
# Release everything when job is finished
video.release()
out.release()
