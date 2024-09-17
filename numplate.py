import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont


model=YOLO('num8.pt')
ocr = PaddleOCR(use_angle_cls=True,lang="korean")
cap = cv2.VideoCapture(0)

font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font = ImageFont.truetype(font_path, 20)
def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")

    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results[0] is not None:
#        print(results)
        for result in results[0]:
            print(result)
            text = result[1][0]
            detected_text.append(text.encode('utf-8').decode('utf-8'))
      
    # Join all detected texts into a single string
    return ''.join(detected_text)
my_file = open("numberplate.txt", "r")
data = my_file.read()
class_list = data.split("\n")
frame_count=0
start_time = time.time()
while True:
    ret,frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (640,640))
    results=model.predict(frame,imgsz=240)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    
    
    for index,row in px.iterrows():

 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        crop = frame[y1:y2, x1:x2]
        crop=cv2.resize(crop,(110,70))
        text = perform_ocr(crop)
        text = text.replace('(', '').replace(')', '').replace(',', '').replace(']','').replace('-',' ')
        print(text)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text((x1 - 50, y1-30), text, font=font, fill=(255,0,0))
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

        #cvzone.putTextRect(frame, f'{text}', (x1 - 50, y1 - 30), 1, 1)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time
    cvzone.putTextRect(frame, f'FPS: {round(fps,2)}', (10,50), 4, 2)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()