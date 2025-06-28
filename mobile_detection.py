from ultralytics import YOLO
import cv2
import pygame
import time

# Initialize pygame mixer for sound
pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm.wav.WAV")

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)
last_alarm_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]
    phone_detected = False

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        # Only detect "cell phone"
        if label == "cell phone" and conf > 0.5:
            phone_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Phone", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Play alarm if phone detected
    if phone_detected and time.time() - last_alarm_time > 1:
        print("Phone detected!")
        alarm.play()
        last_alarm_time = time.time()

    # Show result
    cv2.imshow("Phone Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()