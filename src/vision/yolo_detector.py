import os
import json
import numpy as np

def extract_camera_features_from_images(image_dir=None, zone_id="SGR_ZONE_001"):
    """
    Modular Computer Vision / Object Detection Feature Extractor.
    Processes camera frames or simulates frame object detection (YOLO/CNN)
    and converts detections into tabular features for the Random Forest risk pipeline.
    
    Architecture:
    Camera Image -> YOLO/CNN Detection -> Detection Metadata (Human/Vehicle count, Confidence) -> Tabular Risk Model
    """
    print(f"[CV Feature Extractor] Extracting vision features for camera stream in {zone_id}...")

    # Simulated object detection results from YOLO / Pre-trained CNN model
    detections = [
        {"class": "person", "confidence": 0.94, "bbox": [120, 80, 210, 310], "is_night": True},
        {"class": "person", "confidence": 0.89, "bbox": [250, 95, 315, 290], "is_night": True},
        {"class": "car", "confidence": 0.91, "bbox": [400, 150, 620, 380], "is_night": True},
    ]

    human_count = sum(1 for d in detections if d["class"] == "person")
    vehicle_count = sum(1 for d in detections if d["class"] in ["car", "truck", "motorcycle"])
    avg_confidence = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0
    night_movement_count = sum(1 for d in detections if d["is_night"])

    # Convert to tabular camera features for Random Forest model
    camera_features = {
        "Zone_ID": zone_id,
        "Human_Detection_Count": human_count,
        "Vehicle_Detection_Count": vehicle_count,
        "Avg_Camera_Confidence": round(avg_confidence, 4),
        "Night_Movement_Count": night_movement_count,
        "Camera_Activity_Score": round(min(1.0, (human_count * 1.5 + vehicle_count * 1.0 + night_movement_count * 0.8) / 30.0), 4)
    }

    print(f"[CV Feature Extractor] Extracted tabular vision features for {zone_id}:")
    print(json.dumps(camera_features, indent=2))
    return camera_features

if __name__ == "__main__":
    extract_camera_features_from_images()
