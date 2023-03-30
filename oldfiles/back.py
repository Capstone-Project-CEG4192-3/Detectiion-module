from io import BytesIO
import random
import time
import requests
import torch
import cv2
import numpy as np
import threading
class ParkingLotDetector:
    """
    A class for detecting occupied and free parking spots in a parking lot.
    """
    def __init__(self, model_path, video_path,camera=False,cameraid=1):

        """
        Initializes the ParkingLotDetector object.

        Args:
            model_path (str): The path to the model file.
            video_path (str): The path to the video file.
            camera (bool): Whether to use a camera instead of a video file.
            cameraid (int): The camera ID to use if camera is True.
        """
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
        self.objects = {}
        self.tracked_objects = {}
        
        self.next_id = 0
    
    def updateAPI(self):
        """
        Updates the API with the current parking lot data.
        """
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
        except:
            print(f'>Error Cant send data to API')


    def detect_frame(self, frame):
        """
        Detects parking lot data in the given frame.

        Args:
            frame: The frame to detect data in.

        Returns:
            Tuple: A tuple containing the frame with the parking lot data drawn on it and the detection results.
        """
        # Resize the frame
        frame_resized = cv2.resize(frame, (640, 640))

        # Perform object detection
        results = self.model(frame_resized)

        # Extract the detection data
        detections = results.pandas().xyxy[0]
        self.free = (detections['class'] == 1).sum()
        self.total = len(detections)
        self.dataToPost = {'full': self.total - self.free, 'empty': self.free, 'places': self.total}

        # Draw the detection data on the frame
        for _, detection in detections.iterrows():
            xmin, ymin, xmax, ymax = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']].tolist())
            confidence, state = round(float(detection['confidence']), 2), int(detection['class'])
            cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), self.colorStates[state], 2)
            cv2.putText(frame_resized, f"{self.strStates[state]}:{confidence}%", ((xmax + xmin) // 2, (ymin + ymax) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw the occupancy status on the frame
        cv2.rectangle(frame_resized, (260, 10), (530, 40), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame_resized, f"Free Space: {self.free}/{self.total}", (280, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame_resized, (260, 50), (550, 80), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame_resized, f"Occupied Space: {self.total - self.free}/{self.total}", (265, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw the FPS on the frame
        fps_end = cv2.getTickCount()
        time_elapsed = (fps_end - self.fps_start) / cv2.getTickFrequency()
        fps = self.frame_count / time_elapsed
        cv2.putText(frame_resized, f"{int(fps)}FPS", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

        # Return the resized frame and the detection results
        return frame_resized, detections
    
    def track_boxes(self, detections):
        """
        Tracks the detected boxes by assigning them a unique ID and storing their coordinates.

        Args:
            detections (pandas.DataFrame): The detection results from the YOLOv5 model.
            
        Returns:
            dict: A dictionary containing the tracked objects and their information.
        """
        #f = open('spot.txt', 'w')
        tracked_objects = {}
        
        for _, detection in detections.iterrows():
            xmin, ymin, xmax, ymax = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']].tolist())
            state = int(detection['class'])

            # Calculate the center of the box
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2

            # Check if the box is close to an existing object
            matched_id = None
            for object_id, object_data in self.objects.items():
                object_x, object_y, object_state = object_data
                distance = np.sqrt((center_x - object_x) ** 2 + (center_y - object_y) ** 2)

                #if distance < 50 and state == object_state:
                if distance < 50 :
                    matched_id = object_id
                    break

            # If no matching object was found, assign a new ID
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            # Update the object's data in the dictionary
            self.objects[matched_id] = (center_x, center_y, state)
            stateF=0 if self.strStates[state]== 'Occupied' else 1
            tracked_objects[matched_id] = {'class': self.strStates[state], 'coordinates': (xmin, ymin, xmax, ymax),'matched_id' : matched_id,'state':stateF}
            self.tracked_objects[matched_id] = {'class': self.strStates[state], 'coordinates': (xmin, ymin, xmax, ymax),'matched_id' : matched_id,'state':stateF}

            # Write the box information to the file
            #f.write(f"Box ID: {matched_id}, Class: {self.strStates[state]}, Coordinates: ({xmin}, {ymin}, {xmax}, {ymax})\n")
        f = open('spot.txt', 'w')
        for o in self.objects:
            f.write(f"{o+1}>{self.tracked_objects[o]}\n")
        
        f.close()

        
        return tracked_objects

    def run(self):
        """
        Runs the parking lot detector.
        """
        if __name__=="__main__":
            th=threading.Thread(target=self.updateAPI)
            th.start()
        try:
            motion_threshold = 1000
            while True:
                ret, frame = self.cap.read()
                if self.frame_count%7==0 or 1:
                    self.frameTosend,detectedResults=self.detect_frame(frame)
                    self.track_boxes(detectedResults)
                self.frame_count+=1
                if not self.readyToSend:
                    self.readyToSend=True
                cv2.imshow('Processed Frame with Detections', self.frameTosend)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Release the camera and close all windows
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception as e :
            print(f'end of video or error')
            print(e)
            self.readyToSend=-1

    def testRun(self, ):
        """
        Performs a prediction using the model and passes the detections to the track_boxes method.

        Args:
            test_frame: The frame to perform the prediction on.
        """
        
       

        while True:
                ret, frame = self.cap.read()
                if self.frame_count%7==0 or 1:
                    test_frame_resized = cv2.resize(frame, (640, 640))
                    results = self.model(test_frame_resized)
                    detections = results.pandas().xyxy[0]
                    # Track the detected boxes and draw the box information on the frame
                    det=self.track_boxes(detections)## !!! utiliser ca pour dessiner et plot les id aussi 
                   # for _, detection in detections.iterrows():
                    for detection in det:
                        

                        xmin, ymin, xmax, ymax = det[detection]['coordinates']
                        state =det[ detection]['state']
                        cv2.rectangle(test_frame_resized, (xmin, ymin), (xmax, ymax), self.colorStates[state], 2)
                        cv2.putText(test_frame_resized, f"ID : {det[detection]['matched_id']} > {self.strStates[state]}", (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colorStates[state], 2)


                self.frame_count+=1
                if not self.readyToSend:
                    self.readyToSend=True
                cv2.imshow('Processed Frame with Detections',test_frame_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        # Release the camera and close all windows
        self.cap.release()
        cv2.destroyAllWindows()


files=['test.mp4','video.mp4','test1.mp4','test2.mp4','test.avi',]
#files=['video.mp4','test1.mp4','test2.mp4','test.avi',]
randomchoice=random.randint(0,len(files)-1)
for file in files:
    #p=ParkingLotDetector('best.onnx',file)
    p=ParkingLotDetector('best.pt',file)
    #p.run()
    p.testRun()





