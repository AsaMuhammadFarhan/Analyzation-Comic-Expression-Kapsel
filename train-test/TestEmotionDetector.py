#untuk akses live camera dan read video content
import cv2
#untuk numeric operation
import numpy as np
from keras.models import model_from_json

#list emotion index
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# loading semua training data, storing ke loaded_model_json
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

#konversi all model to json
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture("C:\\Users\\Krisna\\Pictures\\Camera Roll\\WIN_20220616_20_13_36_Pro.mp4")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()

    #preprocesing to resize frame to 1280x720px
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    #detect face in video
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    #converting to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    # x dan y adalah start position, w dan h adalah gambar utk membuat kotak
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        # cropping image
        roi_gray_frame = gray_frame[y:y + h, x:x + w]

        # preprocessing cropping image sesuai data training
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)

        # persentasi max yang didapatkan dr nilai emosi
        maxindex = int(np.argmax(emotion_prediction)) 

        # tampilin emosi di kotak
        print(emotion_dict[maxindex])
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    #displaying output image dan emosi ke video
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release all resources
cap.release()
cv2.destroyAllWindows()
