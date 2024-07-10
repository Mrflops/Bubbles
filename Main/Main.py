from flask import Flask, render_template, Response
import cv2
import pupil_apriltags as apriltag
import numpy as np

app = Flask(__name__)

# Known parameters
file_path = 'Calibration/settings.txt'
KNOWN_TAG_SIZE = None
FOCAL_LENGTH = None

with open(file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    if 'FOCAL_LENGTH' in line:
        FOCAL_LENGTH = float(line.split('=')[1].strip())
    elif 'KNOWN_TAG_SIZE' in line:
        KNOWN_TAG_SIZE = float(line.split('=')[1].strip())

if KNOWN_TAG_SIZE == 0:
    print("Make sure to use calibrate.py in the Calibration folder before using!")
else:
    def draw_tag(image, corners, tag_id, distance):
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(image, f"ID: {tag_id}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Dist: {distance:.2f}m", (corners[0][0], corners[0][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def estimate_distance(corners):
        width1 = np.linalg.norm(corners[0] - corners[1])
        width2 = np.linalg.norm(corners[1] - corners[2])
        width3 = np.linalg.norm(corners[2] - corners[3])
        width4 = np.linalg.norm(corners[3] - corners[0])
        average_width = (width1 + width2 + width3 + width4) / 4.0
        distance = (KNOWN_TAG_SIZE * FOCAL_LENGTH) / average_width
        return distance

    def generate_frames():
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Error: Unable to open camera")

        detector = apriltag.Detector()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray)

            for tag in tags:
                corners = tag.corners.astype(int)
                tag_id = tag.tag_id
                distance = estimate_distance(corners)
                draw_tag(frame, corners, tag_id, distance)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/april_tag_detector')
    def april_tag_detector():
        return render_template('april_tag_detector.html')

    @app.route('/training')
    def training():
        return render_template('training.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
