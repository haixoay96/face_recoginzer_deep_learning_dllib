

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image,ImageDraw
import face_recognition
import random
import thread
from face_classification.src.emotionPredict import predictEmotion

def predict(path):
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)
    campare_endcodings = face_recognition.face_encodings(image)
    cv2.imwrite('./test/'+ str(random.randint(0,1000)) + '.png',image)
    list = []
    for index in range(len(face_locations)):
        list.append((face_locations[index], campare_endcodings[index]))
    return list
def predictFromCam(small_rgb_ ):
    face_locations = face_recognition.face_locations(small_rgb_)
    campare_endcodings = face_recognition.face_encodings(small_rgb_,face_locations)
    face_landmarks_list = face_recognition.face_landmarks( small_rgb_, face_locations)
    listpair = []
    for index in range(len(face_locations)):
       listpair.append((face_locations[index], campare_endcodings[index], face_landmarks_list[index]))
    return listpair

def load_train(path):
    print 'Prepare data!'
    for image_path in path:
        head, tail = os.path.split(image_path)
        name = tail.split('_')[0]
        print name
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image,face_locations)
        listTrain.append((name,encodings[0]))
        listEndcode.append(encodings[0])
        print 'Preparing ' + image_path
        top, right, bottom, left = face_locations[0]
        print face_locations
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.imwrite('./check_training/'+tail,image)
    print 'Done Prepare!'
    print 'Ready!'
def predictName(listCampare):
    persons = []
    personsAndList = []
    for index in range(len(listCampare)):
        if(listCampare[index] < 0.65):
            try:
                index_ = persons.index(listTrain[index][0])
                personsAndList[index_][1].append(listCampare[index])
            except ValueError:
                persons.append(listTrain[index][0])
                personsAndList.append((listTrain[index][0], [listCampare[index]]))
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
listEndcode = []
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
    small_frame = cv2.resize(rgb_image,(320,240))
    small_rgb = cv2.resize(rgb_image,(320,240))
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray_image,(320,240))

    i=i+1
    listFace = predictFromCam(small_rgb )
    for (location, endcode, face_landmarks) in listFace:
        top, right, bottom, left = location
        em = ('',0)
        try:
            em = predictEmotion(gray_image,(left*2,top*2,right*2-left*2,bottom*2-top*2))
            print em
        except:
            print 'loi'


        cv2.rectangle(frame, (left*2, top*2), (right*2, bottom*2), (0, 0, 255), 2)
        result = face_recognition.face_distance(listEndcode,endcode)
        (name,per) = predictName(result)
        cv2.putText(frame, name + ' is ' + em[0],
        (left*2, top*2), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        print location
        print 'Predict:' + name
        print per

        # pil_image = Image.fromarray(small_frame)
        # d = ImageDraw.Draw(pil_image, 'RGBA')
        # # Make the eyebrows into a nightmare
        # d.line(face_landmarks['chin'], fill=(68, 54, 39, 150), width=1)
        # d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        # d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        # d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=1)
        # d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=1)
        #             # Gloss the lips
        # d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        # d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        # d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        # d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)
        #
        # # Sparkle the eyes
        # d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        # d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
        #
        # # Apply some eyeliner
        # d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        # d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
        # small_frame = np.array(pil_image)
        # small_frame = rgb_image = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGRA)
        #print result
    #cv2.imwrite('./test/'+ str(random.randint(0,10000)) + '.png',frame)
    #small_frame = rgb_image = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGRA)
    cv2.imshow('Smile Detector',frame)
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
