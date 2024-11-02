# Main.py

import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

###################################################################################################
def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return                                                          # and exit program
    # end if

    imgOriginalScene  = cv2.imread("LicPlateImages/1.png")               # open image

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return                                          # and exit program
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

        cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file

    # end if else

    cv2.waitKey(0)					# hold windows open until user presses a key

    return
# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
   # cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function

###################################################################################################
if __name__ == "__main__":
    main()

----------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

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
        self.root.title("License Plate Detection")
        
        self.btnBrowse = tk.Button(root, text="Browse Image", command=self.browse_image)
        self.btnBrowse.pack(pady=20)
        
        self.lblImage = tk.Label(root)
        self.lblImage.pack()

        self.lblResult = tk.Label(root, text="", font=("Helvetica", 16))
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

    def process_image(self, filepath):
        imgOriginalScene = cv2.imread(filepath)
        
        if imgOriginalScene is None:
            messagebox.showerror("Error", "Image not read from file")
            return
        
        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)
        
        if len(listOfPossiblePlates) == 0:
            self.lblResult.config(text="No license plates were detected")
        else:
            listOfPossiblePlates.sort(key=lambda plate: len(plate.strChars), reverse=True)
            licPlate = listOfPossiblePlates[0]
            cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
            cv2.imshow("imgThresh", licPlate.imgThresh)

            
            if len(licPlate.strChars) == 0:
                self.lblResult.config(text="No characters were detected")
            else:
                drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
                writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
                cv2.imwrite("imgOriginalScene.png", imgOriginalScene)
                self.display_image("imgOriginalScene.png")
                self.lblResult.config(text=f"License plate read from image = {licPlate.strChars}")

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((500, 400), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.lblImage.config(image=img)
        self.lblImage.image = img

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

   # cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
   # cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

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
-------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

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
        self.root.title("License Plate Detection")
        self.root.configure(bg="#f0f0f0")  # Set background color

        # Browse Button
        self.btnBrowse = tk.Button(root, text="Browse Image", command=self.browse_image,
                                   bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.btnBrowse.pack(pady=10)

        # Capture from Webcam Button
        self.btnWebcam = tk.Button(root, text="Capture from Webcam", command=self.capture_from_webcam,
                                   bg="#008CBA", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.btnWebcam.pack(pady=10)

        # Process Video Button
        self.btnVideo = tk.Button(root, text="Process Video", command=self.process_video,
                                  bg="#FF5733", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.btnVideo.pack(pady=10)

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
            if len(licPlate.strChars) == 0:
                return False
            else:
                drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
                writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
                cv2.imwrite("imgOriginalScene.png", imgOriginalScene)
                self.display_image("imgOriginalScene.png")
                self.lblResult.config(text=f"License plate read from image = {licPlate.strChars}")
                return True

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
    #.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

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
--------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

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
        self.root.title("License Plate Detection")
        self.root.configure(bg="#f0f0f0")  # Set background color

        # Browse Button
        self.btnBrowse = tk.Button(root, text="Browse Image", command=self.browse_image,
                                   bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.btnBrowse.pack(pady=10)

        # Capture from Webcam Button
        self.btnWebcam = tk.Button(root, text="Capture from Webcam", command=self.capture_from_webcam,
                                   bg="#008CBA", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.btnWebcam.pack(pady=10)

        # Process Video Button
        self.btnVideo = tk.Button(root, text="Process Video", command=self.process_video,
                                  bg="#FF5733", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
        self.btnVideo.pack(pady=10)

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
            if len(licPlate.strChars) == 0:
                return False
            else:
                drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
                writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
                cv2.imwrite("imgOriginalScene.png", imgOriginalScene)
                self.display_image("imgOriginalScene.png")
                self.lblResult.config(text=f"License plate read from image = {licPlate.strChars}")
                return True

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
    #.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

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
















