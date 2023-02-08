import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, enforce_detection=False, actions=['emotion'])
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(frame, result['dominant_emotion'], (0,50), font, 2, (0,255,0), 2, cv2.LINE_4);
    
    cv2.imshow('Result', frame)
    
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
cap.release()
cv2.destroyAllWindows()