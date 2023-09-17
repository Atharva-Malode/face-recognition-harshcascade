import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

dataset_path = "./data/"
facedata = []
labels = []

classId = 0
nameMap = {}

# Load data from .npy files
for f in os.listdir(dataset_path):
    if f.endswith('.npy'):
        nameMap[classId] = f[:-4]
        dataItem = np.load(dataset_path + f)
        m = dataItem.shape[0]
        facedata.append(dataItem)

        target = classId * np.ones((m,))
        classId += 1
        labels.append(target)

XT = np.concatenate(facedata, axis=0)
yT = np.concatenate(labels, axis=0).astype(int)

# Create and train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(XT, yT)

# Function to predict the class of a new face using KNN
def predict_class_knn(X, classifier):
    return classifier.predict([X])[0]

# Capture video from the camera
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
offset = 20

while True:
    success, img = cam.read()
    if not success:
        print("Your camera is not working")

    faces = model.detectMultiScale(img, 1.3, 5)

    for f in faces:
        x, y, w, h = f
        print(f)

        cropped_face = img[y - offset : y + h + offset, x - offset : x + w + offset]
        cropped_face = cv2.resize(cropped_face, (100, 100))
        flattened_face = cropped_face.flatten()

        classPredicted = predict_class_knn(flattened_face, knn_classifier)
        namePredicted = nameMap[classPredicted]
        print(namePredicted)

        # Display name
        cv2.putText(
            img, namePredicted, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Prediction Window", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
