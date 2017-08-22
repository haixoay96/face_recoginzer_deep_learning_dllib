

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import face_recognition
import random
import thread

def predict(path):
    image = face_recognition.load_image_file(path)
    face_locations, campare_endcodings = face_recognition.face_encodings(image)
    cv2.imwrite('./test/'+ str(random.randint(0,1000)) + '.png',image)
    list = []
    for index in range(len(face_locations)):
        list.append((face_locations[index], campare_endcodings[index]))
    return list
def predictFromCam(image):
    #face_locations = face_recognition.face_locations(image)
    face_locations,campare_endcodings = face_recognition.face_encodings(image)
    list = []
    for index in range(len(face_locations)):
       list.append((face_locations[index], campare_endcodings[index]))
    return list

def load_train(path):
    print 'Prepare data!'
    for image_path in path:
        head, tail = os.path.split(image_path)
        name = tail.split('_')[0]
        print name
        image = face_recognition.load_image_file(image_path)
        face_locations,encodings = face_recognition.face_encodings(image)
        listTrain.append((name,encodings[0]))
        print 'Preparing ' + image_path
    print 'Done Prepare!'
    print 'Ready!'
def predictName(list):
    persons = []
    personsAndList = []
    for (name, result) in list:
        try:
            index = persons.index(name)
            personsAndList[index][1].append(result)
        except ValueError:
            persons.append(name)
            personsAndList.append((name, [result]))
    personsAndAverages = []
    for (persons, kqs) in personsAndList:
        personsAndAverages.append((persons, sum(kqs)/len(kqs)))
    personsAndAverages.sort(key=lambda tup: tup[1])
    if len(personsAndAverages) > 0:
        return personsAndAverages[0]
    return ('unknow', 0.0)


path_know = './know'
path_unknow = './unknow'
train_paths = [os.path.join(path_know, f) for f in os.listdir(path_know)]
compare_paths = [os.path.join(path_unknow, f) for f in os.listdir(path_unknow)]
listTrain = []
# load data train
load_train(train_paths)
cap = cv2.VideoCapture(0)

cap.set(3,1920)
cap.set(4,1080)
i = 0

while True:
    ret, frame = cap.read() # Capture frame-by-frame
    frame = cv2.flip(frame,1)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = frame.shape
    small = cv2.resize(rgb_image,(320,240))
    print i
    i=i+1
    listFace = predictFromCam(small)
    for (location, endcode) in listFace:
        top, right, bottom, left = location
        cv2.rectangle(small, (left, top), (right, bottom), (0, 0, 255), 2)
        result = face_recognition.face_distance(listTrain,endcode)
        (name,per) = predictName(result)
        cv2.putText(small, name,
        (left, top), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        print location
        print 'Predict:' + name
        print per
        #print result
    cv2.imwrite('./test/'+ str(random.randint(0,10000)) + '.png',frame)
    cv2.imshow('Smile Detector',cv2.resize(small,(640,480)))
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
        print predictName(result)
#    cv2.imwrite('./test/'+ str(random.randint(0,1000)) + '.png',image)
