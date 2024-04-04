from flask import Flask, redirect, request, Response, render_template, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import easyocr
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)

reader = easyocr.Reader(['en'])
model = YOLO('best.pt')

class_list = []
with open("coco1.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

area = [(10, 300), (10, 400), (1010, 400), (1010, 300)]

processed_numbers = set()

car_plate_data = []  # Initialize an empty list to store car plate data

def process_video(file_path):
    global car_plate_data  # Access the global car_plate_data variable

    cap = cv2.VideoCapture(file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        frame_height, frame_width, _ = frame.shape
        half_frame_height = frame_height // 2
        half_frame_width = frame_width // 2
        area_width = frame_width // 2

        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            d = int(row[5])
            c = class_list[d]
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
            if result >= 0:
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 10, 20, 20)

                result = reader.readtext(gray)
                if result:
                    text = result[0][1]
                    if text not in processed_numbers:
                        processed_numbers.add(text)
                        with open("car_plate_data.txt", "a") as file:
                            file.write(f"{text}\n")
                        car_plate_data.append(text)  # Append new plate data to the list

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def update_car_plate_data():
    global car_plate_data  # Access the global car_plate_data variable

    while True:
        with open("car_plate_data.txt", "r") as file:
            car_plate_data = file.readlines()  # Read the contents of car_plate_data.txt
        time.sleep(1)  # Wait for 1 second before reading again

@app.route('/')
def index():
    return render_template('upload_video.html')

@app.route('/process_video', methods=['POST'])
def process_video_route():
    if 'video' not in request.files:
        return redirect(request.url)

    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)

    file_path = 'static/' + secure_filename(file.filename)
    file.save(file_path)

    # Start a new thread to continuously update car_plate_data
    update_thread = threading.Thread(target=update_car_plate_data)
    update_thread.start()

    return render_template('process.html', filename=file_path)

@app.route('/video_feed')
def video_feed():
    filename = request.args.get('filename')
    return Response(process_video(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plate_data')
def plate_data():
    global car_plate_data  # Access the global car_plate_data variable
    return "\n".join(car_plate_data)  # Return car plate data as a newline-separated string

if __name__ == '__main__':
    app.run(debug=True)
