import cv2
import numpy as np
import pickle

# database connection
conn = sqlite3.connect('facedata.db')

# declare camera & face detection classifier
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
if not video.isOpened():
    print("Cannot access camera")
    exit()

def add_face(id, first, last, face):
    with conn:
        conn.execute('''
               INSERT INTO faces(student_id, first_name, last_name, face_data)
               VALUES (?,?,?,?);
               ''', (id,first,last, sqlite3.Binary(pickle.dumps(face)))
               )
    conn.commit()

faces_data = []
id = input("Enter your student id: ")
first = input("Enter your first name: ")
last = input("Enter your last name: ")
i = 0

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        
        resized_img = cv2.resize(frame[y:y+h, x:x+w], (50, 50))

        if len(faces_data) < 10 and i % 10 == 0:
            faces_data.append(resized_img)

        i=i+1

        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.line(frame, (x, y), (x + w // 4, y), (255, 255, 255), 2)  
        cv2.line(frame, (x, y), (x, y + h // 4), (255, 255, 255), 2)  
        cv2.line(frame, (x + w, y), (x + w - w // 4, y), (255, 255, 255), 2)  
        cv2.line(frame, (x + w, y), (x + w, y + h // 4), (255, 255, 255), 2)  
        cv2.line(frame, (x, y + h), (x + w // 4, y + h), (255, 255, 255), 2)  
        cv2.line(frame, (x, y + h), (x, y + h - h // 4), (255, 255, 255), 2)  
        cv2.line(frame, (x + w, y + h), (x + w - w // 4, y + h), (255, 255, 255), 2)  
        cv2.line(frame, (x + w, y + h), (x + w, y + h - h // 4), (255, 255, 255), 2)

    k = cv2.waitKey(1)
    cv2.imshow("frame", frame)
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# convery into numpy array
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)

# add the face to database
add_face(id, first, last, faces_data)

conn.close()