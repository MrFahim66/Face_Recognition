import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

path = 'Dataset'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)

for cls in myList:
    currentImage = cv2.imread(f'{path}/{cls}')
    images.append(currentImage)
    classNames.append(os.path.splitext(cls)[0])
print("\n>>>  Encoding Dataset\n\n")


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print(">>>  Encoding Completed!!")


# getting the information in "Information.csv" file

def getInformation(name):
    with open('Information.csv', 'r+') as f:
        DataList = f.readlines()
        nameList = []
        # print(DataList)
        for line in DataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateString}')


# capturing video

cap = cv2.VideoCapture(0)


while True:
    Frame, img = cap.read()
    image_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resizing image
    image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

    faces_Current_Frame = face_recognition.face_locations(image_small)  # getting current frames location
    encode_Current_Frame = face_recognition.face_encodings(image_small, faces_Current_Frame)  # encoding current frames

    for encodeFace, faceLocation in zip(encode_Current_Frame,
                                        faces_Current_Frame):  # looping through both lists to  get the faces distance
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            getInformation(name)

    cv2.imshow('Camera', img)
    cv2.waitKey(1)
