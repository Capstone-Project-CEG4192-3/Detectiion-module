# Parking Lot Detector

This project is a Python script that uses the YOLOv5 object detection model to detect the occupancy of parking spaces in a video or camera stream. It draws boxes around occupied and free parking spaces and displays the number of each in real-time. The script also sends data to an API with the number of total, free, and occupied parking spaces, along with an image of the current frame with detections.
## Requirements
Python 3
PyTorch
OpenCV
Ultralytics' YOLOv5 implementation
Requests
## Installation
Clone the repository to your local machine and install the dependencies using pip:
bash
> git clone https://github.com/username/parking-lot-detector.git cd parking-lot-detector pip install -r requirements.txt
## Usage
The script can be run using a video file or a camera stream.
### Video
To run the script using a video file, provide the path to the file as an argument to the ParkingLotDetector constructor:
> p = ParkingLotDetector('path/to/model.pt', 'path/to/video.mp4') p.run()
### Camera
To run the script using a camera stream, set the camera argument to True and optionally provide the camera ID using the cameraid argument:

> p = ParkingLotDetector('path/to/model.pt', camera=True, cameraid=0) p.run()
## API
The script also sends data to an API with the number of total, free, and occupied parking spaces, along with an image of the current frame with detections. The API endpoint can be configured in the updateAPI method of the ParkingLotDetector class:

>link = 'http://example.com/api/endpoint'

To enable sending data to the API, run the script and start a separate thread for the updateAPI method:

>if __name__ == "__main__": p = ParkingLotDetector('path/to/model.pt', 'path/to/video.mp4') api_thread = threading.Thread(target=p.updateAPI) api_thread.start() p.run()

## Acknowledgements
 This project uses the YOLOv5 object detection model implementation by Ultralytics.
 
ParkAid 

Uottawa 2022-2023
