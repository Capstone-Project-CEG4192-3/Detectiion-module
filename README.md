## Parking Lot Detector

This is a Python-based program that detects and counts the number of occupied and free parking spaces in a parking lot. It uses the YOLOv5 object detection model and processes images captured from a camera or a video file. The program can also send data to an API for further processing.

## Features
Detects occupied and free parking spots in real-time
Supports both video files and camera input
Uses YOLOv5 for object detection
Updates and sends parking data to an API
Provides the option to track objects in the parking lot
Can retrieve images from a remote source using web scraping
## Dependencies
OpenCV
PyTorch
Requests
BeautifulSoup4
NumPy
## Usage
First, install the required packages using pip:


pip install -r requirements.txt

To use the program, you need to initialize the ParkingLotDetector class with the following parameters:

model_path: The path to the model file (YOLOv5 model)
video_path: The path to the video file
camera: Whether to use a camera instead of a video file (optional, default: False)
cameraid: The camera ID to use if camera is True (optional, default: 1)
Then, call the run() method to start the parking lot detector:


p = ParkingLotDetector('best.pt', 'test.mp4')
p.run()
To run the parking lot detector with a remote source, call the runRemoteSource() method:


p = ParkingLotDetector('best.pt', 'test.mp4')
p.runRemoteSource()
## Example

files=['test.mp4','video.mp4','test1.mp4','test2.mp4','test.avi']
randomchoice=random.randint(0,len(files)-1)
for file in files:
    p = ParkingLotDetector('best.pt', file)
    p.runRemoteSource()
This example creates a new ParkingLotDetector instance with a random video file from the files list and runs the detector using a remote image source.
