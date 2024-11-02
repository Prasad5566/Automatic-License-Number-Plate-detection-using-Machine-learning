import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sqlite3
import tkinter.font as tkFont

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

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Indian License Plate Detection")
        self.root.configure(bg="#3E4A89")  # Set background color to a modern dark blue

        # Title Label
        self.lblTitle = tk.Label(root, text="Automatic License Plate Number Detection System", font=("Helvetica", 24, "bold"), bg="#3E4A89", fg="white")
        self.lblTitle.pack(pady=20)

        # Main Frame
        self.main_frame = tk.Frame(root, bg="#3E4A89")
        self.main_frame.pack(pady=20)

        # Button Frame
        self.button_frame = tk.Frame(self.main_frame, bg="#3E4A89")
        self.button_frame.grid(row=0, column=0, padx=20, pady=20)

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

        # About Button
        self.btnAbout = tk.Button(self.button_frame, text="About", command=self.show_about,
                                  bg="#FFD700", fg="black", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.btnAbout.grid(row=0, column=3, padx=10)

        # Displayed Images Frame
        self.display_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.display_frame.grid(row=1, column=0, padx=20, pady=20)

        # Original Image Label
        self.lblOriginalImage = tk.Label(self.display_frame, bg="#f0f0f0")
        self.lblOriginalImage.grid(row=0, column=0, padx=10)

        # Plate Image Label
        self.lblPlateImage = tk.Label(self.display_frame, bg="#f0f0f0")
        self.lblPlateImage.grid(row=0, column=1, padx=10)

        # Threshold Image Label
        self.lblThreshImage = tk.Label(self.display_frame, bg="#f0f0f0")
        self.lblThreshImage.grid(row=0, column=2, padx=10)

        # Result Label
        self.lblResult = tk.Label(root, text="", font=("Helvetica", 16), bg="#3E4A89", fg="white")
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
            #cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
            #scv2.imshow("imgThresh", licPlate.imgThresh)
            if len(licPlate.strChars) == 0:
                return False
            else:
                drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
                writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
                cv2.imwrite("imgOriginalScene.png", imgOriginalScene)
                self.display_image("imgOriginalScene.png", licPlate.imgPlate, licPlate.imgThresh)
                self.lblResult.config(text=f"License plate read from image = {licPlate.strChars}")

                # Fetch RTO Information
                rto_info = self.fetch_rto_info(licPlate.strChars)
                self.display_result(f"{licPlate.strChars}", rto_info)
                return True

    def fetch_rto_info(self, license_plate):
        conn = sqlite3.connect('rto_info.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rto_info WHERE license_plate=?", (license_plate,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return f"Owner: {row[1]}, Vehicle Model: {row[2]}, Registration Date: {row[3]}"
        else:
            return "No information available"

    def display_image(self, original_path, plate_img, thresh_img):
        # Display original image
        img_original = Image.open(original_path)
        img_original = img_original.resize((500, 400), Image.LANCZOS)
        img_original = ImageTk.PhotoImage(img_original)
        self.lblOriginalImage.config(image=img_original)
        self.lblOriginalImage.image = img_original

        # Display plate image
        img_plate = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        img_plate = img_plate.resize((250, 200), Image.LANCZOS)
        img_plate = ImageTk.PhotoImage(img_plate)
        self.lblPlateImage.config(image=img_plate)
        self.lblPlateImage.image = img_plate

        # Display threshold image
        img_thresh = Image.fromarray(cv2.cvtColor(thresh_img, cv2.COLOR_BGR2RGB))
        img_thresh = img_thresh.resize((250, 200), Image.LANCZOS)
        img_thresh = ImageTk.PhotoImage(img_thresh)
        self.lblThreshImage.config(image=img_thresh)
        self.lblThreshImage.image = img_thresh

    def display_result(self, license_plate, vehicle_info):
        # Set up a font for bold text
        bold_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
        
        result_text = f"License plate number = {license_plate}\nVehicle information: {vehicle_info}"
        
        self.lblResult.config(text=result_text, font=bold_font)

    def show_about(self):
        self.about_window = tk.Toplevel(self.root)
        self.about_window.title("About ANPR")
        self.about_window.configure(bg="#3E4A89")

        lblAboutTitle = tk.Label(self.about_window, text="About ANPR", font=("Helvetica", 18, "bold"), bg="#3E4A89", fg="white")
        lblAboutTitle.pack(pady=20)

        about_text = ("Automatic Number Plate Recognition (ANPR) is a technology that uses optical character "
                      "recognition on images to read vehicle registration plates. It can use existing closed-circuit "
                      "television, road-rule enforcement cameras, or cameras specifically designed for the task. "
                      "ANPR is used by police forces around the world for law enforcement purposes, including to "
                      "check if a vehicle is registered or licensed. It is also used for electronic toll collection "
                      "on pay-per-use roads and as a method of cataloging the movements of traffic, for example by highways agencies.")

        lblAboutText = tk.Label(self.about_window, text=about_text, font=("Helvetica", 14), bg="#3E4A89", fg="white", wraplength=600)
        lblAboutText.pack(pady=20)

        btnClose = tk.Button(self.about_window, text="Close", command=self.about_window.destroy,
                             bg="#FF5733", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        btnClose.pack(pady=20)

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    p2fRectPoints = np.intp(p2fRectPoints)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

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
