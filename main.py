# IMPORT

## Import modul Interface
import pygame
import py_button
from pyPath import path

## Import modul Interval
from time import time, sleep
import threading

## Import modul Model
import cv2
import numpy as np
from keras.models import model_from_json

# -----

# INTERFACE

## Variabel buat display window
screenWidth = 600
screenHeight = 500
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption('Interface from python')

## Comic Image
comic0 = pygame.image.load(path + 'Comic0.png')
comic0 = pygame.transform.scale(comic0, (400, 400))
comic1 = pygame.image.load(path + 'Comic1.jpg')
comic1 = pygame.transform.scale(comic1, (400, 400))
comic2 = pygame.image.load(path + 'Comic2.jpg')
comic2 = pygame.transform.scale(comic2, (400, 400))
comic3 = pygame.image.load(path + 'Comic3.jpg')
comic3 = pygame.transform.scale(comic3, (400, 400))
comic4 = pygame.image.load(path + 'Comic4.jpg')
comic4 = pygame.transform.scale(comic4, (400, 400))
comic5 = pygame.image.load(path + 'Comic5.jpg')
comic5 = pygame.transform.scale(comic5, (400, 400))
comic6 = pygame.image.load(path + 'Comic6.jpg')
comic6 = pygame.transform.scale(comic6, (400, 400))
comic7 = pygame.image.load(path + 'Comic7.jpg')
comic7 = pygame.transform.scale(comic7, (400, 400))
comic8 = pygame.image.load(path + 'Comic8.jpg')
comic8 = pygame.transform.scale(comic8, (400, 400))
comic9 = pygame.image.load(path + 'Comic9.jpg')
comic9 = pygame.transform.scale(comic9, (400, 400))
comic10 = pygame.image.load(path + 'Comic10.jpg')
comic10 = pygame.transform.scale(comic10, (400, 400))
index = 0
comicArray = [comic0, comic1, comic2, comic3, comic4, comic5, comic6, comic7, comic8, comic9, comic10]

## Button - with image prop w="362" h="126"
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

## Array Temporary
expressionFinalData = []
pageFinalData = []
temporaryData = []

def temporaryLogging(expression):
  if expression:
    print("Ada expression kok: " + expression)
  if expression == "Neutral":
    print("Lah kok Neutral")
  temporaryData.append(expression)


def finalLogging(page):
  angryCount = temporaryData.count("Angry")
  disgustedCount = temporaryData.count("Disgusted")
  fearfulCount = temporaryData.count("Fearful")
  happyCount = temporaryData.count("Happy")
  # neutralCount = temporaryData.count("Neutral")
  sadCount = temporaryData.count("Sad")
  surprisedCount = temporaryData.count("Surprised")

  print("Count: " + str(angryCount) + " " + str(disgustedCount) + " " + str(fearfulCount) + " " + str(happyCount) + " " + str(sadCount) + " " + str(surprisedCount) + " "   )

  summary = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
  indexSummary = 0
  x = angryCount
  if disgustedCount > x:
    x = disgustedCount
    indexSummary = 1
  if fearfulCount > x:
    x = fearfulCount
    indexSummary = 2
  if happyCount > x:
    x = happyCount
    indexSummary = 3
  # if neutralCount > x:
  #   x = neutralCount
  #   indexSummary = 4
  if sadCount > x:
    x = sadCount
    indexSummary = 5
  if surprisedCount > x:
    x = surprisedCount
    indexSummary = 6
  
  dataLength = len (temporaryData)

  if dataLength == 0:
    print('Temporary Kosong Tikus')

  expressionFinalData.append(summary[indexSummary])
  pageFinalData.append(page)
  temporaryData.clear()

# -----

# MODEL
output = ["Neutral"]
runModel = True
def model():
  ## List emotion index
  emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

  ## Loading semua training data, storing ke loaded_model_json
  json_file = open('model/emotion_model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()

  ## Konversi all model to json
  emotion_model = model_from_json(loaded_model_json)

  # Load weights into new model
  emotion_model.load_weights("model/emotion_model.h5")
  print("Loaded model from disk")

  # start the webcam feed
  cap = cv2.VideoCapture(0)

  while runModel:
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
      output.append(emotion_dict[maxindex])

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if runModel == False:
        break

  # Release all resources
  cap.release()
  cv2.destroyAllWindows()

t1 = threading.Thread(target=model)
t1.start()

# -----

# THREAD INTERVAL

runInterval = True
def interval():
  print('Auto creating log every 1 second')
  while runInterval:
    sleep(1 - time() % 1)
    temporaryLogging(output[-1])
    # print('[ ===== Creating Logs ===== ]')

t2 = threading.Thread(target=interval)
t2.start()

# -----

# OUTPUT
## Ganti nama orang ketika ganti orang
personName = "AsaMuhammadFarhan" # Minimalisir penggunaan ada spasi
fileName = "ExpressionLogExsys.txt"
def finalResult():
  f = open(fileName, "a")
  f.writelines(personName + "'s expression:\n")
  for x in range(len(pageFinalData)):
    f.writelines("Hal " + str(pageFinalData[x]) + ": " + expressionFinalData[x] + "\n")
  f.writelines("\n\n")
  f.close()
# -----

# THREAD INTERFACE

runInterface = True
while runInterface:
    screen.fill((0, 0, 0))
    
    # Button callback
    if backButton.draw(screen):
        if index != 0:
          finalLogging(index)
          print('Trigger Back, sekarang halaman', index + 1)
          index = index-1

    if nextButton.draw(screen):
        finalLogging(index)
        print('Trigger Next, sekarang halaman', index + 1)
        if (index != len(comicArray) - 1):
          index = index+1
        if index == len(comicArray) - 1:
          finalResult()

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
            runModel = False

    pygame.display.update()

pygame.quit()
runInterface = False
runInterval = False
runModel = False
t1.join()
t2.join()

# -----