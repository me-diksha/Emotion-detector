import cv2                                                                                                              # used to get the live feed from cameras and to read video content
import numpy as np                                                                                                      # to do numeric operation
from keras.models import model_from_json                                                                                 # to load model which we stored in json model file after training

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
# we created dictionary to map our value with emotion acc to dic

# load json and create model
json_file = open('model/emotional_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)                                                                            # jo bhi neural network ka model humne store kra h json file ma use hum uha use kr rhe hai

# after making model we  are applying load and weights into new model
emotion_model.load_weights("model/emotion_model.h5")                                                                          # we are loading weights from our weights stored in h5 into our new model

# to get input from cameras
# cap = cv2.VideoCapture(0)


# for using it on video database
cap= cv2.VideoCapture("C:\\Users\\lenovo\\Downloads\\production_id_4098429 (2160p).mp4")   # here we can pass video path
#"
# "C:\\Users\\lenovo\\Videos\\emotion sample\\mixkit-excited-girl-talking-on-video-call-with-her-cell-phone-8745-medium.mp4"
# "C:\\Users\\lenovo\\Videos\\emotion sample\\mixkit-sad-and-desperate-girl-crying-and-screaming-25589-medium.mp4"
# "C:\\Users\\lenovo\\Videos\\emotion sample\\pexels-alena-darmel-6654092 (2160p).mp4"
#"C:\\Users\\lenovo\\Downloads\\production_id_4101230 (2160p).mp4"
while True:
    ret, frame = cap.read()
    if frame is not None and not frame.size == 0:
        frame = cv2.resize(frame, (1200, 720))
        # Rest of your processing on the resized frame
    else:
        print("Frame is empty or None. Cannot resize.")
    # to resize our data into laptop screen
    if not ret:
        break
    face_detect = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')                                               #give harcascade file path     harcascade make frame around the face and pass that into emotional model to dtetect face expressions
    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGRA2GRAY)                                                                   # whatever the image we are reading we are converting it into a gray scale as our model is trained for gray scale images we dont pass rgb images as it cxan gibe miscallenous result


    # detect all faces available on camera
    num_faces = face_detect.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    #to take each face available on camera and to detect them
    for(x, y, w, h) in num_faces:   # to access each face x ,y,width and height to provide position of each face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0))                                                              # it help in making rectangle around images faces with the help of x,y coordinate width and height
        roi_gray_frame = gray_frame[y:y+h, x:x+w]                                                                           # with this we are croping area of ur interest and storing it in roi_gray_frame
        crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)                 #now what ever ccrop image we are getting we are resizing it into our trai  images dimension

        # now we will predict the emotion
        emotion_prediction = emotion_model.predict(crop_img)
        maxindex = int(np.argmax(emotion_prediction))
        print(maxindex)# percentage of each emotion confidenec will be passed as index and compared with emotion_dict
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2 , cv2.LINE_AA) # perdiction ka frame bta rhe h uske compare ke bad result put text se use frame pa likhwa rge and font se font bta rhe coordinate se text ki location


    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break      # just showing our output image with emotion


cap.release()    #release of all resorces used
cv2.destroyAllWindows()


                                       #working good but slow :(





