import cv2
import numpy as np
import pyttsx3
import threading
import time
from datetime import datetime

cap = cv2.VideoCapture(0)


engine = pyttsx3.init()
speaking = False 
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

report_data = []

def generate_summary(x, y, w, h):
    region = "upper-left" if x < 320 and y < 240 else \
             "upper-right" if x >= 320 and y < 240 else \
             "lower-left" if x < 320 and y >= 240 else "lower-right"

    width_cm = round(w * 0.0264, 1)
    height_cm = round(h * 0.0264, 1)

    summary = (
        f"A potentially abnormal region has been identified in the {region} section "
        f"of the scanned frame. The region spans approximately {width_cm} centimeters "
        f"in width and {height_cm} centimeters in height. Further clinical analysis is "
        f"recommended to confirm the findings."
    )
    return summary, region, width_cm, height_cm

def speak_alert(summary):
    global speaking
    speaking = True
    time.sleep(1.2) 
    engine.say(summary)
    engine.runAndWait()
    speaking = False

def analyze_frame(frame):
    global speaking

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (5, 5))
    edged = cv2.Canny(closed, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tumor_detected = False
    tumor_mask = np.zeros_like(gray)
    summary_spoken = False
    frame_height, frame_width = frame.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if area < 1000 or w < 20 or h < 20 or w > 0.9 * frame_width or h > 0.9 * frame_height:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0 or area / hull_area < 0.5:
            continue

        tumor_detected = True
        cv2.drawContours(tumor_mask, [contour], -1, 255, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not speaking and not summary_spoken:
            summary, region, width_cm, height_cm = generate_summary(x, y, w, h)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            report_data.append({
                'time': timestamp,
                'region': region,
                'width_cm': width_cm,
                'height_cm': height_cm
            })

            threading.Thread(target=speak_alert, args=(summary,), daemon=True).start()
            summary_spoken = True

    colored_mask = cv2.cvtColor(tumor_mask, cv2.COLOR_GRAY2BGR)
    overlay = frame.copy()
    overlay[tumor_mask == 255] = (0, 0, 255)

    alpha = 0.4
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    if tumor_detected:
        cv2.putText(frame, "Tumor Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No Tumor Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Analyzed Frame", frame)
    cv2.waitKey(0)
    cv2.destroyWindow("Analyzed Frame")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    cv2.imshow('Tumor Detection - Press C to Capture', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        frozen_frame = frame.copy()
        analyze_frame(frozen_frame)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if report_data:
    with open("tumor_detection_report.txt", "a") as f:
        f.write("Tumor Detection Report\n")
        f.write("======================\n\n")
        for entry in report_data:
            f.write(f"Time: {entry['time']}\n")
            f.write(f"Region: {entry['region']}\n")
            f.write(f"Width: {entry['width_cm']} cm\n")
            f.write(f"Height: {entry['height_cm']} cm\n\n")

    print("✅ Report saved as 'tumor_detection_report.txt'")
else:
    print("No detections — no report generated.")
