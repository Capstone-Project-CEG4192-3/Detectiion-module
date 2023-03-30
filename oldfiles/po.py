import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import urllib.request

def display_first_image(url):
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
            
            # Display the image
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Aucune balise <img> trouvée.")
    else:
        print(f"Erreur lors de la récupération de la page Web. Code d'état: {response.status_code}")

webpage_url = "http://www.insecam.org/en/view/996923/"
display_first_image(webpage_url)
