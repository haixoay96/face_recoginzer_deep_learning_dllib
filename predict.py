

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import face_recognition
import random


def predict(path):
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)
    campare_endcodings = face_recognition.face_encodings(image)
    cv2.imwrite('./test/'+ str(random.randint(0,1000)) + '.png',image)
    list = []
    for index in range(len(face_locations)):
        list.append((face_locations[index], campare_endcodings[index]))
    return list
def predictFromCam(image):
    face_locations = face_recognition.face_locations(image)
    campare_endcodings = face_recognition.face_encodings(image)
    list = []
    for index in range(len(face_locations)):
        list.append((face_locations[index], campare_endcodings[index]))
    return list


path_know = './know'
path_unknow = './unknow'
train_paths = [os.path.join(path_know, f) for f in os.listdir(path_know)]
compare_paths = [os.path.join(path_unknow, f) for f in os.listdir(path_unknow)]
listTrain = []
print 'Prepare data!'
for image_path in train_paths:
    image = face_recognition.load_image_file(image_path)
    biden_encoding = face_recognition.face_encodings(image)[0]
    listTrain.append(biden_encoding)
    print 'Preparing ' + image_path
print 'Done Prepare!'
print 'Ready!'

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
while True:

    ret, frame = cap.read() # Capture frame-by-frame
    frame = cv2.flip(frame,1)
    image_pil = Image.fromarray(frame).convert('RGB')
    rgb = np.array(image_pil, 'uint8')
    height, width, channels = frame.shape
    print height
    print width
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    small = cv2.resize(frame,(320,180))
#    equ = cv2.equalizeHist(gray)
    listFace = predictFromCam(small)
    for (location, endcode) in listFace:
        top, right, bottom, left = location
        cv2.rectangle(frame, (left*(width/320), top*(height/180)), (right*(width/320), bottom*(height/180)), (0, 0, 255), 2)
        result = face_recognition.face_distance(listTrain,endcode)
        print location
        print result
    cv2.imwrite('./test/'+ str(random.randint(0,1000)) + '.png',frame)
    cv2.imshow('Smile Detector', rgb)
    c = cv2.cv.WaitKey(7) % 0x100
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

for image_path in compare_paths:
    print 'Predict: '+ image_path
    image_pil = Image.open(image_path).convert('L')
    # Convert the image format into numpy array
    image = np.array(image_pil, 'uint8')
    listFace = predict(image_path)
    for (location, endcode) in listFace:
        top, right, bottom, left = location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        result = face_recognition.face_distance(listTrain,endcode)
        print location
        print result
#    cv2.imwrite('./test/'+ str(random.randint(0,1000)) + '.png',image)
