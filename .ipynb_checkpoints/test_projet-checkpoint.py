
import cv2 as cv
import numpy as np
import time 
import psutil
import sys
import ipywidgets as widgets
from IPython.display import display

from echelle import *
from dance import *

import torch
import ctypes
from Arm_Lib import Arm_Device
Arm = Arm_Device()
s_time=1000
angleBras=85
Delay = 20
compteurDetect = 0

#TEST
#labels=['waldo']

#global
model = 'General'
Detect = 0
Pointage = 1
posActuelle = 0

# Create layout
button_layout  = widgets.Layout(width='200px', height='70px', align_self='center')
# Output printing
output = widgets.Output()
# Detection button
detec_button = widgets.Button(description='Detection', button_style='success', layout=button_layout)
# Exit button
exit_button = widgets.Button(description='Exit', button_style='danger', layout=button_layout)
# Image control
imgbox = widgets.Image(format='jpg', height=480, width=640, layout=widgets.Layout(align_self='center'))
# Vertical placement
controls_box = widgets.VBox([imgbox,exit_button,detec_button], layout=widgets.Layout(align_self='center'))
# ['auto', 'flex-start', 'flex-end', 'center', 'baseline', 'stretch', 'inherit', 'initial', 'unset']
                
def exit_button_Callback(value):
    global model
    model = 'Exit'
    print("Click exit")
#     with output: print(model)
exit_button.on_click(exit_button_Callback)

def detec_button_Callback(value):
    global Detect
    Detect=1
    print("Click detection")
detec_button.on_click(detec_button_Callback)


def observation() : # Mise en position d'observation
    Arm.Arm_serial_servo_write6_array([angleBras, 125, 0, 0, 90, 30],s_time)
    
def distanceFromY(y): # Déduire la distance à laquelle pointer en fonction de la distance de l'objet en pixel (y)
    Donnees = [0,0,0,14,34,65,90,117,145,176,212,245,269,302,339,378,413,461,480,480]
    i=0
    while (y > Donnees[i] and i<25):
        i += 1
    return (i-1)

def moveBras():
    global imageG
    global angleBras
    global Detect
    global compteurDetect
    global Pointage
    
    global labels
    global coor
    
    
    # Initialisation et analyse d'image
    (lowerb, upperb) = ((149,112,104),(201,255,255))
    color_mask = imageG.copy()
    hsv_img = cv.cvtColor(imageG, cv.COLOR_BGR2HSV)
    color = cv.inRange(hsv_img, lowerb, upperb)
    color_mask[color == 0] = [0, 0, 0]
    gray_img = cv.cvtColor(color_mask, cv.COLOR_RGB2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst_img = cv.morphologyEx(gray_img, cv.MORPH_CLOSE, kernel)
    ret, binary = cv.threshold(dst_img, 10, 255, cv.THRESH_BINARY)
    contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # python3
    
    for i, cnt in enumerate(contours):
        mm = cv.moments(cnt)
        if mm['m00'] == 0:
            continue
        cx = mm['m10'] / mm['m00']
        cy = mm['m01'] / mm['m00']
        area = cv.contourArea(cnt)
        approx = cv.approxPolyDP(cnt,0.02*cv.arcLength(cnt, True),True) # Nombre de côtés du polynome détecté
        
        # SI C'EST UN CERCLE (8 côtés ou plus)
        if Pointage == 1 and len(labels)>0:
            
            #n = len(labels)
            #for i in range(n):
            x_shape, y_shape = imageG.shape[1], imageG.shape[0]
            row = coor[0]
            if row[4] >= 0.2:
                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                x = np.int((x1+x2)/2)
                y = np.int((y1+y2)/2)
                
                imageG = cv.circle(imageG,(x,y),5,(0,255,0),-1)
            #on centre le cercle sur l'image = on pivote le bras pour qu'il soit pile en face
            if x<310*2:
                if x<160:
                    angleBras+=5
                    Arm.Arm_serial_servo_write(1, angleBras, s_time)
                    time.sleep(0.5)
                else :
                    angleBras += 1
                    Arm.Arm_serial_servo_write(1, angleBras, s_time)
                    time.sleep(1)
                compteurDetect -=1
            elif x>330:
                if x<480:
                    angleBras-=5
                    Arm.Arm_serial_servo_write(1, angleBras, s_time)
                    time.sleep(0.5)
                else :
                    angleBras -= 1
                    Arm.Arm_serial_servo_write(1, angleBras, s_time)
                    time.sleep(1)
                compteurDetect -=1
            else:
                if compteurDetect >= 2 :
                    allerADistance(Arm, distanceFromY(y), angleBras+1)
                    time.sleep(5)
                    observation()
                    Detect = 0
                else :
                    compteurDetect += 1
            return 1
        
        # SI Waldo
        if len(labels) >= 1  : 
            Arm.Arm_Buzzer_On(1)
            
        # SI C'EST UN TRIANGLE
        if area > 500.0 and len(approx) == 3 : 
            dance(Arm)
    return 0


def goToPositionSuivante(): # Mouvement périodique du bras (il pivote pour parcourir tout son environnement)
    global posActuelle
    global Delay
    global angleBras
    positions = [85,125,160,125,85,45,10,45]
    posActuelle = (posActuelle+1)%8
    angleBras = positions[posActuelle]
    Arm.Arm_serial_servo_write6_array([angleBras, 125, 0, 0, 90, 30],s_time)
    Delay=20


class ObjectDetection :
    

    """
    Class implemets Yolo5 model to make inferences on a camera video using OpenCv
    """

    def __init__(self,out_file,camera=0):
        """
        Initializes the class with youtube url and output file.
        param out_file: A valid output filename format .avi
        param camera: if you have several camera, you can specify the device. By default, it's 0.
        """
        #init selon nb_image +1 
        self.camera = camera
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file= out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)
    
    
    def load_model(self):
        """
        Load Yolo5 pretrained model from local, a model trained with a notebook on google colab.
        """
        model = torch.hub.load('./yolov5', 'custom', path='yolov5/runs/train/exp/weights/best_1.pt', source='local',force_reload=True)  # local repo
        #model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
        return model
    
    def score_frame(self, frame):
        """
        Takes a single frame as input, and swores the frame using yolo5 model
        param frame: input frame in numpy/list/tuple format.
        return: Labels and Coordinates of objects detected by model in the frame
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord =results.xyxyn[0][:,-1], results.xyxyn[0][:,:-1]
        return labels, cord
    
    def class_to_label(self, x):
        """
        For a given label vlau, return corresponding strin label
        :param x: numerical label
        return corresponding label
        """
        return self.classes[int(x)]
    
    def plot_boxes(self,results, frame):
        """
        takes a frame and its results as input, plots the bounding boxes an label on to the frame
        param results: contains labels ands coordinates pridcted by model on the given frame
        param frame: Frame which has been scored.
        return frame with bounding boxes, labels and conf ploted on it
        """
        labels, cord = results 
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                bgr = [0,255,0]
                cv.rectangle(frame, (x1,y1), (x2,y2) ,bgr, 2)
                cv.putText(frame, f"{self.class_to_label(labels[i])} {row[4]}" , (x1,y1), cv.FONT_HERSHEY_SIMPLEX, 0.9,(bgr),2,cv.LINE_AA)
        return frame
    
    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by fram, and write the output into a new file
        """
        
        
        global imageG
        global labels
        global Delay
        global coor
        
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            sys.exit()
            
        print('open camera')
        x_shape=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        y_shape=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        #four_cc = cv.VideoWriter_fourcc(*"MJPG")
        #out = cv.VideoWriter(self.out_file, four_cc, 40 ,(x_shape,y_shape))
        
        while cap.isOpened():
            try:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                frame = cv.resize(frame, (640, 480))
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                results = self.score_frame(frame)
                labels, coor = results
                ##self.save_image(frame,results)
                frame = self.plot_boxes(results, frame)
                imageG = frame
                if Detect == 1 or Pointage == 1:
                    k = moveBras()
                    if k==0 and Delay<0:
                        goToPositionSuivante()
                    else :
                        Delay -= 1
                if model == 'Exit':
                    model=='General'
                    cv.destroyAllWindows()
                    cap.release()
                    sys.exit()
                    break
                imgbox.value = cv.imencode('.jpg', imageG)[1].tobytes()
                #out.write(frame)
            except KeyboardInterrupt:
                capture.release()


#detection = ObjectDetection("video_test_projet.avi")
#detection()

def test_detection():
    global imageG
    global labels
    global Delay

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit()

    print('open camera')
    x_shape=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    y_shape=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    #four_cc = cv.VideoWriter_fourcc(*"MJPG")
    #out = cv.VideoWriter(self.out_file, four_cc, 40 ,(x_shape,y_shape))

    while cap.isOpened():
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv.resize(frame, (640, 480))
            imageG = frame
            if Detect == 1:
                k = moveBras()
                if k==0 and Delay<0:
                    goToPositionSuivante()
                else :
                    Delay -= 1
            if model == 'Exit':
                cv.destroyAllWindows()
                cap.release()
                sys.exit()
                break
            imgbox.value = cv.imencode('.jpg', imageG)[1].tobytes()
            #out.write(frame)
        except KeyboardInterrupt:
            capture.release()
    