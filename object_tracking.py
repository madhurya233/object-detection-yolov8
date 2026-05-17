import cv2
from ultralytics import YOLO
import os

# ----------------------------
# Load YOLO model (pretrained COCO)
# ----------------------------
model = YOLO("yolov8n.pt")  # nano version, fast for CPU

# ----------------------------
# Folder containing media files
# ----------------------------
media_folder = "video"
# Accept mp4 videos and image formats including avif
media_files = [f for f in os.listdir(media_folder) if f.lower().endswith((".mp4", ".jpg", ".png", ".avif"))]

if not media_files:
    print("No images or videos found in folder!")
    exit()

    

# ----------------------------
# Process each file
# ----------------------------
for file in media_files:
    file_path = os.path.join(media_folder, file)
    print(f"Processing: {file}")

    # ----------------------------
    # If image
    # ----------------------------
    if file.lower().endswith((".jpg", ".png", ".avif")):
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Cannot open image: {file}")
            continue

        results = model(frame)
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(f"Detection - {file}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ----------------------------
    # If video
    # ----------------------------
    elif file.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Cannot open video: {file}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            for result in results:
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(f"Detection - {file}", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to skip video
                break

        cap.release()
        cv2.destroyAllWindows()

print("All files processed!")

