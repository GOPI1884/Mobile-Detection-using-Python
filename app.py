from flask import Flask, render_template
import threading

app = Flask(__name__)

# ✅ This is your updated start_detection() function
def start_detection():
    import cv2
    import pygame
    from ultralytics import YOLO

    # Initialize alarm
    pygame.mixer.init()
    pygame.mixer.music.load('alarm.wav.WAV')

    # Load YOLO model
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run detection on the frame
        results = model.predict(frame, save=False, verbose=False)

        detected = False  # Reset detection for this frame

        # Draw bounding boxes & check detections
        for r in results:
            boxes = r.boxes
            names = r.names
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # Get box coordinates
                x1, y1, x2, y2 = xyxy

                # Draw bounding box & label
                label = f"{cls_name} {confidence:.2f}"
                color = (0, 255, 0)  # Green box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Check if it's a cell phone
                if cls_name == 'cell phone':
                    print(f"Detected: {cls_name}")
                    detected = True

        # Play alarm if phone is detected
        if detected:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)  # -1 means loop continuously
        else:
            pygame.mixer.music.stop()

        # Show webcam feed
        cv2.imshow("Driver Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ✅ Flask routes (MUST be added)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection')
def start():
    threading.Thread(target=start_detection).start()
    return "Driver detection started! Check your webcam window."


# ✅ Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
