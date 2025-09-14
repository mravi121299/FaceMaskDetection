


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np 
import argparse
import cv2
import os
from plyer import notification
from playsound import playsound
import tkinter
from tkinter import messagebox
import smtplib

root =tkinter.Tk()
root.withdraw()



def detect_and_predict_mask(frame, faceNet, maskNet):
    
    (h, w) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227),
        (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)
    
    faces =[]
    locs =[] 
    preds =[]


    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box =detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")


            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))


            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)


            faces.append(face)
            locs.append((startX, startY, endX, endY))


    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)



prototxtPath = r"C:\project\caffe\deploy.prototxt"
weightsPath = r"C:\project\caffe\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)



maskNet = load_model(r'C:\project2\my_mask_detector_Model.model')



vidcap = cv2.VideoCapture(0)
SUBJECT="Subject"
TEXT="One visiter violated Face Mask Policy.See in the camera to recognize user.A person is detected without Mask"


while True:
    success,frame = vidcap.read()
    try:
        frame = cv2.resize(frame , (400,400),interpolation=cv2.INTER_AREA)   
    except:
        break
    
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask" 
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # if mask < withoutMask:
           # path=os.path.abspath("Alarm.wav")
            #playsound(path)
            
        #if mask < withoutMask:
           # message='Subject:{}\n\n{}'.format(SUBJECT,TEXT)
           # mail=smtplib.SMTP('smtp.gmail.com',587)
            #mail.ehlo()
           # mail.starttls()
           # mail.login("ravirmishra2995898@gnail.com","raviPass")
           # mail.sendmail('ravirmishra2995898@gnail.com','ravimishra121299@gnail.com',message)
           # mail.close
    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vidcap.release()
cv2.destroyAllWindows() 





