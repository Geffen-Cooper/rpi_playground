import cv2
from PIL import Image

cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)
cap.set(cv2.CAP_PROP_FPS, 36)
while True:
    # read frame
    ret, image = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")
    image = cv2.rotate(image,cv2.ROTATE_180)
    cv2.imshow('img',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


