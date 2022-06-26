
# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# penskalaan ulang gambar 1./255 untuk data test dan training
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# melakukan preprocessing data train
train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# melakukan preprocessing data test
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# membuat model CNN untuk training data
emotion_model = Sequential()

#input shape menyesuaikan target size 48px dan 1 (grayscale)
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))

#membuat layer cnn baru, 64px dgn kernel size 3 dan activation relu
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

#add ke pool 2,2
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

#untuk menghindari noise, menggunakan dropout
emotion_model.add(Dropout(0.25))

#membuat layer CNN dengan 128px, mengulang hal yg diatas
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

#untuk mendatarkan nilai
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))

#mendapatkan 7 klasifikasi dr emosi
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

#menambahkan loss function, optimizer dengan learning rate 0.0001 dan matrics sebagai akurasi
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# data yang tadi di preprocess, di training
emotion_model_info = emotion_model.fit_generator(
        #data di variable train generator
        train_generator,

        #steps per epoch adalah jumlah gambar dibagi 64
        steps_per_epoch=28709 // 64,
        epochs=100,

        #validasi untuk data testing
        validation_data=validation_generator,

        #jumlah data dibagi 6
        validation_steps=7178 // 64)

# semua model structure disimpan di json file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# menyimpan data emotion model di file tersebut
emotion_model.save_weights('emotion_model.h5')

