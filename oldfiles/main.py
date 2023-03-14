
'''
This code is an implementation of a parking space occupancy detection system using YOLOv5 object detection model. 
It captures frames from a video or camera feed, applies object detection to detect vehicles in the frames, 
counts the number of free and occupied parking spaces, and sends this information along with the frame to a REST API endpoint.

The main code block runs a loop that reads frames from the video feed, applies object detection every 7 frames, 
counts free and occupied parking spaces, draws bounding boxes around detected vehicles, and displays the results on the frame. 
The loop also prepares the information to be sent to the API endpoint and puts the frame in a global variable for the API thread to access. 
The API thread runs concurrently to send the prepared information and image to the API endpoint.

'''
#imports
from io import BytesIO
import time
import requests
import torch
import cv2
import numpy as np
import threading


model = torch.hub.load('ultralytics/yolov5','custom', path='best.onnx', )



#test videos
files=['test.mp4','test.avi','test1.mp4','test2.mp4','video.mp4']
file=files[-1]

# Open the Video
cap=cv2.VideoCapture(file)

# Open the camera
#cap = cv2.VideoCapture(1)

#variables
fps_start = cv2.getTickCount()
frame_count = 0
colorStates=[(0,0,255),(0,255,0)]
strStates=['Occupied','Free']
total=0
free=0
dataToPost={'full': total-free, 'empty': free, 'places': total}
readyToSend=False

#thread function to send infos to API  (ie. dict with space count and image of the space)
def updateAPI():
    global dataToPost
    global readyToSend
    try:
     while True:
            time.sleep(0.5)
            if readyToSend:
                readyToSend=False
                link='http://mdakk072.pythonanywhere.com/status'
                #link='http://localhost:5000/status'
                retval, buffer = cv2.imencode('.jpg', frameTosend)
                image_file = BytesIO(buffer)
                files = {'image': image_file}
                r = requests.post(link, data=dataToPost, files=files)
                #print(r.text)
    except:
     print(f'>Error Cant send data to API')


#start API function
if __name__=="__main__":
    th=threading.Thread(target=updateAPI)
    th.start()


#detection Loop
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    frame=cv2.resize(frame,(640,640))
    # Inference every 7 frames to gain some FPS
    if frame_count%7==0:
        results = model(frame)
    frame_count += 1
    fps_end = cv2.getTickCount()
    time_elapsed = (fps_end - fps_start) / cv2.getTickFrequency()
    fps = frame_count / time_elapsed
    # Display  FPS on image
    cv2.putText(frame, f"{int(fps)}FPS", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

    # Get detections
    pd=results.pandas().xyxy[0]
    free=len([x  for x in pd['class'] if x==1] )
    total=len(pd)

    #draw boxes on detections 
    for index,row in pd.iterrows():
        xmin,ymin,xmax,ymax,precisiion,state=int(row['xmin']),int(row['ymin']),int(row['xmax']),int(row['ymax']),round(float(row['confidence']),2),int(row['class'])
        #print(xmin,ymin,xmax,ymax,precisiion,state)
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), colorStates[state], 2)
        #cv2.putText(frame, f"{strStates[state]}:{precisiion}%", ((xmax+xmin)//2,(ymin+ ymax)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    #if not already sending put frame to send in global variable
    if not readyToSend:
     frameTosend=frame
     resultsTosend=results
     dataToPost={'full': total-free, 'empty': free, 'places': total}
     readyToSend=True


    #write infos on frame
    cv2.putText(frame, f"FREE SPACE : {free}/{int(total)}", (280, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 255), 1)
    cv2.putText(frame, f"OCCUPIED SPACE : {total-free}/{int(total)}", (280, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 1)
    cv2.imshow('Processed Frame with Detections', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# end
