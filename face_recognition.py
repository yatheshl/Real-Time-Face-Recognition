import face_recognition
import cv2
import numpy as np
import os
from sklearn import svm

KNOWN_FACES_DIR = 'known_faces'

known_face_encodings = []
known_face_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    encodings = []
    if name.startswith('.'):  # skip hidden files
        continue
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        if not filename.endswith('.jpg') and not filename.endswith('.jpeg') and not filename.endswith('.png'):
                continue

        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        face_encoding = face_recognition.face_encodings(image)[0]
        encodings.append(face_encoding)
    if len(encodings) > 0:
        known_face_encodings.append(np.mean(encodings, axis=0))
        known_face_names.append(name)
        

# Train SVM classifier
X_train = known_face_encodings
y_train = known_face_names
clf = svm.SVC(kernel='linear', C=1, probability=True)
clf.fit(X_train, y_train)

video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
face_accuracy = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame, model = 'hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_accuracy = []
        for face_encoding in face_encodings:
            # Predict using SVM classifier
            face_encoding = np.array(face_encoding).reshape(1, -1)
            name_prob = clf.predict_proba(face_encoding)[0]
            name = clf.predict(face_encoding)[0]
            accuracy = round(name_prob[np.where(clf.classes_ == name)[0][0]] * 100, 2)
            face_names.append(name)
            face_accuracy.append(accuracy)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name, accuracy in zip(face_locations, face_names, face_accuracy):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"{name} ({accuracy}%)"
        cv2.putText(frame, text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()