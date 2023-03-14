from io import BytesIO
import random
import time
import requests
import torch
import cv2
import numpy as np
import threading


class ParkingLotDetector:
    def __init__(self, model_path, video_path,camera=False,cameraid=1):
        self.model = torch.hub.load('ultralytics/yolov5','custom', path=model_path)
        self.cap = cv2.VideoCapture(video_path)
        if camera:
            self.cap = cv2.VideoCapture(cameraid)
        self.fps_start = cv2.getTickCount()
        self.frame_count = 0
        self.colorStates=[(0,0,255),(0,255,0)]
        self.strStates=['Occupied','Free']
        self.total=0
        self.free=0
        self.dataToPost={'full': self.total-self.free, 'empty': self.free, 'places': self.total}
        self.readyToSend=False
        self.frameTosend=None

    def updateAPI(self):
        try:
            while True:
                time.sleep(0.5)
                if self.readyToSend== True:
                    link='http://mdakk072.pythonanywhere.com/status'
                    #link='http://localhost:5000/status'
                    retval, buffer = cv2.imencode('.jpg', self.frameTosend)
                    image_file = BytesIO(buffer)
                    files = {'image': image_file}
                    r = requests.post(link, data=self.dataToPost, files=files)
                    self.readyToSend=False
                if self.readyToSend==-1:
                    return
                

                    #print(r.text)
        except:
            print(f'>Error Cant send data to API')
    
    def detect_frame(self, frame):

        frame = cv2.resize(frame, (640, 640))
        results = self.model(frame)
        pd = results.pandas().xyxy[0]
        self.free = len([x for x in pd['class'] if x == 1])
        self.total = len(pd)
        self.dataToPost={'full': self.total-self.free, 'empty': self.free, 'places': self.total}

        for index, row in pd.iterrows():
            xmin, ymin, xmax, ymax, precision, state = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(
                row['ymax']), round(float(row['confidence']), 2), int(row['class'])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.colorStates[state], 2)
            cv2.putText(frame, f"{self.strStates[state]}:{precision}%", ((xmax + xmin) // 2, (ymin + ymax) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # Free space text with background rectangle
        cv2.rectangle(frame, (260, 10), (530, 40), (0, 255, 0), cv2.FILLED)
        #cv2.putText(frame, f"Free Space: {self.free}/{int(self.total)}", (265, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Free Space: {self.free}/{int(self.total)}", (280, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        # Occupied space text with background rectangle
        cv2.rectangle(frame, (260, 50), (550, 80), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, f"Occupied Space: {self.total - self.free}/{int(self.total)}", (265, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        fps_end = cv2.getTickCount()
        time_elapsed = (fps_end - self.fps_start) / cv2.getTickFrequency()
        fps = self.frame_count / time_elapsed
        cv2.putText(frame, f"{int(fps)}FPS", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
        return frame, results
    
    def run(self):

        if __name__=="__main__":
            th=threading.Thread(target=self.updateAPI)
            th.start()
        try:
            while True:
                ret, frame = self.cap.read()

                if self.frame_count%7==0 or 1:
                    self.frameTosend,detectedResults=self.detect_frame(frame)
                self.frame_count+=1

                if not self.readyToSend:
                    self.readyToSend=True

                cv2.imshow('Processed Frame with Detections', self.frameTosend)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            # Release the camera and close all windows
            self.cap.release()
            cv2.destroyAllWindows()
        except:
            print('end of video or error')
            self.readyToSend=-1

files=['test.mp4','video.mp4','test1.mp4','test2.mp4','test.avi',]


randomchoice=random.randint(0,len(files)-1)
#file=files[3]
#file=files[randomchoice]
### To use Camera (coment the loop)
#p=ParkingLotDetector('best.pt',file,camera=True)
#p.run()

for file in files:
    p=ParkingLotDetector('best.pt',file)
    #p=ParkingLotDetector('best.onnx',file)
    p.run()