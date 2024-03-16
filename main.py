import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from datetime import datetime
import easyocr

reader = easyocr.Reader(['en'])

model = YOLO('best1.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('mycarplate.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

# area = [(27, 417), (16, 456), (1015, 451), (992, 417)]



count = 0
list1 = []
processed_numbers = set()

# Open file for writing car plate data
with open("car_plate_data.txt", "a") as file:
    file.write("NumberPlate\tDate\tTime\n")  # Writing column headers

while True:    
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
   
    frame = cv2.resize(frame, (1020, 500))
    frame_height, frame_width, _ = frame.shape
    half_frame_height = frame_height // 2
    half_frame_width = frame_width // 2
    area_width = frame_width // 2

    area = [(half_frame_width - area_width // 2, half_frame_height),
        (half_frame_width - area_width // 2, 0),
        (half_frame_width + area_width // 2, 0),
        (half_frame_width + area_width // 2, half_frame_height)]

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
                  list1.append(text)
                  current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                  with open("car_plate_data.txt", "a") as file:
                      file.write(f"{text}\t{current_datetime}\n")
                      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                      cv2.imshow('crop', crop)

      
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()    
cv2.destroyAllWindows()
