from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Run on webcam (press 'q' to quit)
results = model.predict(source=0, show=True, stream=True)

for r in results:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
