import cv2
import matplotlib.pyplot as plt

def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

#Initiate the camera
cap1 = cv2.VideoCapture(-1)

#create image plot
im1 = plt.imshow(grab_frame(cap1))

plt.ion()

while True:
    im1.set_data(grab_frame(cap1))
    plt.pause(0.2)