

from io import BytesIO
import random
import time
from bs4 import BeautifulSoup
import requests
import torch
import cv2
import numpy as np
import threading
import urllib.request

class ParkingLotDetector:
    """
    A class for detecting occupied and free parking spots in a parking lot.
    """
    def __init__(self, model_path, video_path, camera=False, cameraid=1):
        """
        Initializes the ParkingLotDetector object.

        Args:
            model_path (str): The path to the model file.
            video_path (str): The path to the video file.
            camera (bool): Whether to use a camera instead of a video file.
            cameraid (int): The camera ID to use if camera is True.
        """

        # Load the model from the given path
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

        # Initialize the video capture object with the given video file
        self.cap = cv2.VideoCapture(video_path)

        # If a camera should be used, reinitialize the video capture object with the camera ID
        if camera:
            self.cap = cv2.VideoCapture(cameraid)

        # Initialize the frame count and FPS counter
        self.fps_start = cv2.getTickCount()
        self.frame_count = 0

        # Define the color states and textual states for occupied and free parking spaces
        self.colorStates = [(0, 0, 255), (0, 255, 0)]
        self.strStates = ['Occupied', 'Free']

        # Initialize parking space counts and data to be sent to an API
        self.total = 0
        self.free = 0
        self.dataToPost = {'full': self.total - self.free, 'empty': self.free, 'places': self.total}
        self.readyToSend = False

        # Initialize the frame to be sent and the frame detections
        self.frameTosend = None
        self.frameDetections = {}

        # Initialize object tracking data
        self.objects = {}
        self.next_id = 0

        # Initialize test data for an external API
        self.APItestData = {'id': None, 'available': None}
        self.testAdrienAPI = False

    def getLocalisation(self):
        ip_address = requests.get('https://api.ipify.org').text
        locationInfos={}
        # Envoie une requête à l'API GeoIP
        url = 'http://ip-api.com/json/' + ip_address
        response = requests.get(url)

        # Analyse la réponse de l'API
        if response.status_code == 200:
            location = response.json()
            

            locationInfos ['country']=location['country']
            locationInfos ['city']=location['city']
            locationInfos['lat'] =location['lat']
            locationInfos ['lon']=location['lon']
            print(  locationInfos)
            return  locationInfos
        else:
            print("Impossible d'obtenir votre localisation.")

    def send_request_to_api(self, data):
        """
        Sends a request to the API with the given parking lot data.

        Args:
            data (dict): The parking lot data to send.
        """
        api_url = "https://park-aid-api.herokuapp.com/parkingSpots/?id=" + data['id'] + "&available=" + data['available']

        try:
            response = requests.post(api_url)
            if response.status_code == 200:

                print("Data successfully sent to the API.")
                print(f'>[ INFO ] Request was sent to API : {api_url} ')
                print(f'>[ INFO ]               with Data  : {data} ')
                print(f'request Object after sending : {response}')
            
            else:
                print(f"Error sending data to the API. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error sending data to the API: {e}")

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
                    self.getLocalisation()
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
            #if confidence<0.5: continue
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
        """! NOT WORKING 
        Tracks the detected boxes by assigning them a unique ID and storing their coordinates.

        Args:
            detections (pandas.DataFrame): The detection results from the YOLOv5 model.
            
        Returns:
            dict: A dictionary containing the tracked objects and their information.
        """
        f = open('spot.txt', 'w')
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

                if distance < 50 and state == object_state:
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

            # Write the box information to the file
            f.write(f"Box ID: {matched_id}, Class: {self.strStates[state]}, Coordinates: ({xmin}, {ymin}, {xmax}, {ymax})\n")
        f.close()
        
        return tracked_objects
    
    def getRemoteImage(self):
        url = "http://www.insecam.org/en/view/996923/"
        url = "http://www.insecam.org/en/view/945438/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find the first <img> tag
            img_tag = soup.find("img")

            if img_tag:
                img_url = img_tag["src"]
                
                # Read the image from the URL
                with urllib.request.urlopen(img_url) as response:
                    image_data = response.read()
                
                # Convert the image data to a numpy array
                image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
                
                # Decode the numpy array into an OpenCV image (BGR format)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                '''
                # Display the image
                cv2.imshow("Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
                return img
            else:
                print("Aucune balise <img> trouvée.")
        else:
            print(f"Erreur lors de la récupération de la page Web. Code d'état: {response.status_code}")
    
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
                self.frame_count+=1
                if not self.readyToSend:
                    self.readyToSend=True

                if not self.testAdrienAPI : 
                    self.testAdrienAPI=True
                    randomTestId=random.randint(5,430)
                    randomTestAvailable=random.randint(0,1)
                    self.APItestData['id']=randomTestId
                    self.APItestData['available']=randomTestAvailable
                    self.send_request_to_api(self.APItestData)
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

    def runRemoteSource(self): 
        while True:
                frame=self.getRemoteImage()
                if self.frame_count%7==0 or 1:
                    self.frameTosend, self.frameDetections=self.detect_frame(frame)
                self.frame_count+=1
                if not self.readyToSend:
                    self.readyToSend=True
                if not self.testAdrienAPI : 
                    self.testAdrienAPI=True
                    randomTestId=random.randint(5,430)
                    randomTestAvailable=random.randint(0,1)
                    self.APItestData['id']=randomTestId
                    self.APItestData['available']=randomTestAvailable
                    self.send_request_to_api(self.APItestData)
                cv2.imshow('Processed Frame with Detections', self.frameTosend)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Release the camera and close all windows
        self.cap.release()
        cv2.destroyAllWindows()


files=['test.mp4','video.mp4','test1.mp4','test2.mp4','test.avi',]
#files=['test1.mp4','test2.mp4','test.avi',]
randomchoice=random.randint(0,len(files)-1)
for file in files:
    #p=ParkingLotDetector('best.onnx',file)
    p=ParkingLotDetector('best.pt',file)
    #p.run()
    p.runRemoteSource()
