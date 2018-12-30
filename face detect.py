import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier("D:\\codes\\face database\\haaarcascade\\haarcascade_frontalface_default.xml")
#img1=cv2.imread("D:\\codes\\face database\\13o.jpg")
#img1=cv2.resize(img1,(500,500))



cap=cv2.VideoCapture(0)
while(1):
     _, frame=cap.read()
     blk1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     face=face_cascade.detectMultiScale(blk1,scaleFactor=1.05,minNeighbors=5)
     for face_co in face:
         x,y,w,h=face_co.reshape(4)
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
         cv2.imshow('Detected',frame)
         key=cv2.waitKey(1)
         if key==ord('q'):
             break
cap.release()
cv2.destroyAllWindows()

    
