# IMPORT

# Import modul Interface
import pygame
import py_button
from pyPath import path

# Import modul Interval
from time import time, sleep
import threading

# Import modul Model
import cv2
import numpy as np
from keras.models import model_from_json

# -----

# INTERFACE

# Variabel buat display window
screenWidth = 800
screenHeight = 500
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption('Interface from python')

# Comic Image
comic1 = pygame.image.load(path + 'Comic1.jpg')
comic2 = pygame.image.load(path + 'Comic2.jpg')
comic3 = pygame.image.load(path + 'Comic3.jpg')
comic4 = pygame.image.load(path + 'Comic4.jpg')
comic5 = pygame.image.load(path + 'Comic5.jpg')
comic1 = pygame.transform.scale(comic1, (400, 300))
comic2 = pygame.transform.scale(comic2, (400, 300))
comic3 = pygame.transform.scale(comic3, (400, 300))
comic4 = pygame.transform.scale(comic4, (400, 300))
comic5 = pygame.transform.scale(comic5, (400, 300))
index = 0
comicArray = [comic1, comic2, comic3, comic4, comic5]

# Button - with image prop w="362" h="126"
backButtonImage = pygame.image.load(path + 'py_back.png').convert_alpha()
nextButtonImage = pygame.image.load(path + 'py_next.png').convert_alpha()
backButton = py_button.Button(
    100,
    450,
    backButtonImage,
    100/362
)
nextButton = py_button.Button(
    (screenWidth - 100) - (nextButtonImage.get_width() * 100/362),
    450,
    nextButtonImage,
    100/362
)

# Array Temporary
expressionFinalData = []
pageFinalData = []
temporaryData = []

def temporaryLogging(expression):
  temporaryData.append(expression)

def finalLogging(page):
  angry = temporaryData.count("Angry")
  disgusted = temporaryData.count("Disgusted")
  fearful = temporaryData.count("Fearful")
  happy = temporaryData.count("Happy")
  neutral = temporaryData.count("Neutral")
  sad = temporaryData.count("Sad")
  surprised = temporaryData.count("Surprised")

  summary = "Angry"
  x = angry
  if disgusted > x:
    x = disgusted
    summary = "Disgusted"
  if fearful > x:
    x = fearful
    summary = "Fearful"
  if happy > x:
    x = happy
    summary = "Happy"
  if neutral > x:
    x = neutral
    summary = "Neutral"
  if sad > x:
    x = sad
    summary = "Sad"
  if Surprised > x:
    x = Surprised
    summary = "Surprised"

  expressionFinalData.append(summary)
  pageFinalData.append(page)
  temporaryData.clear()

# -----

# MODEL
output = ""
def model():
  # List emotion index
  emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

  # Loading semua training data, storing ke loaded_model_json
  json_file = open('model/emotion_model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()

  # Konversi all model to json
  emotion_model = model_from_json(loaded_model_json)

  # Load weights into new model
  emotion_model.load_weights("model/emotion_model.h5")
  print("Loaded model from disk")

  # start the webcam feed
  cap = cv2.VideoCapture(0)

  while True:
      # Find haar cascade to draw bounding box around face
      ret, frame = cap.read()

      # Preprocesing to resize frame to 1280x720px
      frame = cv2.resize(frame, (1280, 720))
      if not ret:
          break

      # Detect face in video
      face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

      # Converting to grayscale
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # Detect faces available on camera
      num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

      # Take each face available on the camera and Preprocess it
      # x dan y adalah start position, w dan h adalah gambar utk membuat kotak
      for (x, y, w, h) in num_faces:
          cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

          # Cropping image
          roi_gray_frame = gray_frame[y:y + h, x:x + w]

          # Preprocessing cropping image sesuai data training
          cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

          # Predict the emotions
          emotion_prediction = emotion_model.predict(cropped_img)

          # Persentasi max yang didapatkan dr nilai emosi
          maxindex = int(np.argmax(emotion_prediction)) 

          # Tampilin emosi di kotak
          print(emotion_dict[maxindex])
          output = emotion_dict[maxindex]
          cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

      # # Show Image Camera, but camera dimatiin
      # cv2.imshow('Emotion Detection', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  #release all resources
  cap.release()
  cv2.destroyAllWindows()

t1 = threading.Thread(target=model)
t1.start()
# -----

# THREAD INTERVAL

runInterval = True
def interval():
  while runInterval:
    sleep(1 - time() % 1)
    temporaryLogging(output)
    print('[ ===== Creating Logs ===== ]')

t2 = threading.Thread(target=interval)
t2.start()

# -----

# THREAD INTERFACE

runInterface = True
while runInterface:
    screen.fill((0, 0, 0))
    
    # Button callback
    if backButton.draw(screen):
        if index != 0:
          finalLogging(index)
          index = index-1
          print('Trigger Back, sekarang halaman', index + 1)

    if nextButton.draw(screen):
        if index != len(comicArray - 1):
          finalLogging(index)
          index = index+1
          print('Trigger Next, sekarang halaman', index + 1)
        if index == len(comicArray - 1):
          print(expressionFinalData)

    screen.blit(
      comicArray[index],
      ((screenWidth/2) - 200, (50))
    )

    # Window event handler
    for event in pygame.event.get():
        # Quit button
        if event.type == pygame.QUIT:
            runInterface = False
            runInterval = False

    pygame.display.update()

pygame.quit()

# -----