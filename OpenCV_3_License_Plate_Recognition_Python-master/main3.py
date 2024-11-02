import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import requests  # Import requests library

import DetectChars
import DetectPlates
import PossiblePlate

# Module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

API_URL = "https://api.example.com/vehicle-info"  # Example API URL
API_KEY = "your_api_key_here"  # Replace with your API key

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detection")
        self.root.configure(bg="#f0f0f0")  # Set background color

        # Button Frame
        self.button_frame = tk.Frame(root, bg="#f0f0f0")
        self.button_frame.pack(pady=10)

        # Browse Button
        self.btnBrowse = tk.Button(self.button_frame, text="Browse Image", command=self.browse_image,
                                   bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.btnBrowse.grid(row=0, column=0, padx=10)

        # Capture from Webcam Button
        self.btnWebcam = tk.Button(self.button_frame, text="Capture from Webcam", command=self.capture_from_webcam,
                                   bg="#008CBA", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.btnWebcam.grid(row=0, column=1, padx=10)

        # Process Video Button
        self.btnVideo = tk.Button(self.button_frame, text="Process Video", command=self.process_video,
                                  bg="#FF5733", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.btnVideo.grid(row=0, column=2, padx=10)

        # Displayed Image Label
        self.lblImage = tk.Label(root, bg="#f0f0f0")
        self.lblImage.pack()

        # Result Label
        self.lblResult = tk.Label(root, text="", font=("Helvetica", 16), bg="#f0f0f0")
        self.lblResult.pack(pady=20)

        self.knn_trained = self.train_knn()

    def train_knn(self):
        blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()
        if not blnKNNTrainingSuccessful:
            messagebox.showerror("Error", "KNN training was not successful")
            return False
        return True

    def browse_image(self):
        if not self.knn_trained:
            return
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if filepath:
            self.process_image(filepath)

    def capture_from_webcam(self):
        if not self.knn_trained:
            return
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Press any key to capture', frame)
            if cv2.waitKey(1) != -1:  # Any key is pressed
                self.process_image_from_array(frame)
                break
        cap.release()
        cv2.destroyAllWindows()

    def process_video(self):
        if not self.knn_trained:
            return
        filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if filepath:
            cap = cv2.VideoCapture(filepath)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('Video Frame', frame)
                if cv2.waitKey(1) != -1:  # Any key is pressed
                    self.process_image_from_array(frame)
                    break
            cap.release()
            cv2.destroyAllWindows()

    def process_image(self, filepath):
        imgOriginalScene = cv2.imread(filepath)
        self.process_image_from_array(imgOriginalScene)

    def process_image_from_array(self, imgOriginalScene):
        if imgOriginalScene is None:
            messagebox.showerror("Error", "Image not read from file")
            return

        detected = self.detect_license_plate(imgOriginalScene)
        if not detected:
            self.lblResult.config(text="No license plates were detected")

    def detect_license_plate(self, imgOriginalScene):
        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

        if len(listOfPossiblePlates) == 0:
            return False
        else:
            listOfPossiblePlates.sort(key=lambda plate: len(plate.strChars), reverse=True)
            licPlate = listOfPossiblePlates[0]
            cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
            cv2.imshow("imgThresh", licPlate.imgThresh)
            if len(licPlate.strChars) == 0:
                return False
            else:
                drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
                writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
                cv2.imwrite("imgOriginalScene.png", imgOriginalScene)
                self.display_image("imgOriginalScene.png")
                self.lblResult.config(text=f"License plate read from image = {licPlate.strChars}")
                self.fetch_vehicle_info(licPlate.strChars)  # Fetch vehicle info using the detected license plate
                return True

    def fetch_vehicle_info(self, license_plate):
        try:
            response = requests.get(API_URL, params={'plate': license_plate, 'api_key': API_KEY})
            data = response.json()
            if response.status_code == 200 and 'vehicle' in data:
                vehicle_info = data['vehicle']
                info_text = f"Owner: {vehicle_info['owner_name']}\nModel: {vehicle_info['model']}\nYear: {vehicle_info['year']}"
                self.lblResult.config(text=info_text)
            else:
                self.lblResult.config(text="Vehicle information not found")
        except Exception as e:
            self.lblResult.config(text=f"Error fetching vehicle information: {str(e)}")

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((500, 400), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.lblImage.config(image=img)
        self.lblImage.image = img

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
   # cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
   # cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)

    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene
    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)
    ptCenterOfTextAreaY = int(intPlateCenterY)

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
