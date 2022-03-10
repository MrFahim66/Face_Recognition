import cv2
import face_recognition

# loading image set from Dataset

data_01 = face_recognition.load_image_file('Dataset/Fahim Ur Rahman.jpg')
data_01 = cv2.cvtColor(data_01, cv2.COLOR_BGR2RGB)
data_02 = face_recognition.load_image_file('Dataset/Md Mehedi Hasan.jpg')
data_02 = cv2.cvtColor(data_02, cv2.COLOR_BGR2RGB)

# getting face location & encoding the face

face_location_01 = face_recognition.face_locations(data_01)[0]
encode_data_01 = face_recognition.face_encodings(data_01)[0]
cv2.rectangle(data_01, (face_location_01[3], face_location_01[0]), (face_location_01[1], face_location_01[2]), (255, 0, 255), 2)

face_location_02 = face_recognition.face_locations(data_02)[0]
encode_data_02 = face_recognition.face_encodings(data_02)[0]
cv2.rectangle(data_02, (face_location_02[3], face_location_02[0]), (face_location_02[1], face_location_02[2]),
              (255, 0, 255), 2)

# comparing both faces & getting the face distance value

results = face_recognition.compare_faces([encode_data_01], encode_data_02)
face_distance = face_recognition.face_distance([encode_data_01], encode_data_02)
cv2.putText(data_01, f'{results} {round(face_distance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255),
            2)
# print(results)
# print(face_distance)

# showing both images with the result

cv2.imshow('Data - 01', data_01)
cv2.imshow('Data - 02', data_02)
cv2.waitKey(0)
