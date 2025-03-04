import shutil
import os
import sys
import cv2
import pygame
import time
from gtts import gTTS
from PIL import Image, ImageTk
from mutagen.mp3 import MP3
import numpy as np
import os
from random import shuffle
from tqdm import \
    tqdm

print('Plant Disease Detection...')
dirPath = "testpicture"
fileList = os.listdir(dirPath)
for fileName in fileList:
    os.remove(dirPath + "/" + fileName)
print('Press  Button-2 to Capture Image')
print('Press "c" to Capture Image')
vs = cv2.VideoCapture(0)
while True:
    ret, image = vs.read()
    if not ret:
        break
    cv2.imshow('Plant Disease', image)
    if cv2.waitKey(1) & 0xFF == ord('c'):
    #if cv2.waitKey(1) & GPIO.input(SW2) == 0:
        cv2.imwrite('testpicture/result.png', image)
        break
vs.release()
cv2.destroyAllWindows()

verify_dir = 'testpicture'
print("path " + verify_dir)
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'plantdiseasedetection-{}-{}.model'.format(LR, '2conv-basic')

def process_verify_data():
    verifying_data = []
    for img in tqdm(os.listdir(verify_dir)):
        path = os.path.join(verify_dir, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        verifying_data.append([np.array(img), img_num])
    np.save('verify_data.npy', verifying_data)
    return verifying_data


verify_data = process_verify_data()

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.compat.v1.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 7, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

import matplotlib.pyplot as plt

fig = plt.figure()

for num, data in enumerate(verify_data):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
    model_out = model.predict([data])[0]
    print(model_out)
    print('model {}'.format(np.argmax(model_out)))

    if np.argmax(model_out) == 0:
        str_label = 'healthy'
    elif np.argmax(model_out) == 1:
        str_label = 'bacterial'
    elif np.argmax(model_out) == 2:
        str_label = 'earlyblight'
    elif np.argmax(model_out) == 3:
        str_label = 'lateblight'
    elif np.argmax(model_out) == 4:
        str_label = 'leafmold'
    elif np.argmax(model_out) == 5:
        str_label = 'septoria'
    elif np.argmax(model_out) == 6:
        str_label = 'yellowcurlvirus'

    if str_label == 'healthy':
        diseasename = "The plant has no disease"
        rem = 'The plant is'
        rem1= 'Healthy'
        load = Image.open('testpicture/result.png')
        english= "The Plant is Healthy "
        kannada= "kannada"

    elif str_label == 'bacterial':
        diseasename = "Bacterial Spot"
        load = Image.open('testpicture/result.png')
        rem = "The remedies for Bacterial Spot are:\n"
        rem1 = " 1.Discard any affected leaf or the complete plant if necessary. \n 2.Use antibiotics such as streptomycin or oxytetracycline."
        english= "The disease recognised is Bacterial spot and The remedies for Bacterial are Discard any affected leaf or the complete plant if necessary and use antibiotics such as streptomycin or oxytetracycline."
        kannada= "kannada"

    elif str_label == 'earlyblight':
        diseasename = "Early Blight"
        load = Image.open('testpicture/result.png')
        rem = "The remedies for Early Blight are:\n"
        rem1 = " 1.Spray the plant with Bonide Liquid with copper fungicide concentrate. \n 2.Solarizing the soil to kill the bacteria before they get to the plants."
        english= "The disease recognised is Early Blight and The remedies for Earlyblight are Spray the plant with Bonide Liquid with copper fungicide concentrate and Solarizing the soil to kill the bacteria before they get to the plants."
        kannada= "kannada"

    elif str_label == 'lateblight':
        diseasename = "Late Blight"
        load = Image.open('testpicture/result.png')
        rem = "The remedies for Late Blight are:\n"
        rem1 = " 1.Discard any affected leaf or the complete plant if necessary. \n 2.Use antibiotics such as streptomycin or oxytetracycline."
        english= "The disease recognised is Late Blight and The remedies for Bacterial are Discard any affected leaf or the complete plant if necessary and use antibiotics such as streptomycin or oxytetracycline."    
        kannada= "kannada"
                    
    elif str_label == 'leafmold':
        diseasename = "Leaf Mold"
        load = Image.open('testpicture/result.png')
        rem = "The remedies for Leaf Mold are:\n"
        rem1 = " 1.Avoid watering plant overhead. \n 2.Use Calcium chloride based sprays."
        english= "The disease recognised is leafmold and The remedies for Leafmold are Avoid watering plant overhead and Use Calcium chloride based sprays."
        kannada= "kannada"

    elif str_label == 'septoria':
        diseasename = "Septoria"
        load = Image.open('testpicture/result.png')
        rem = "The remedies for Septoria are:\n"
        rem1 =  " 1.Remove all fallen plant debris. \n 2.Use copper fungicides."
        english= "The disease recognised is septoria and The remedies for Spectoria are Remove all fallen plant debris and Use copper fungicides"
        kannada= "kannada"

    elif str_label == 'yellowcurlvirus':
        diseasename = "Yellow Curl Virus"
        load = Image.open('testpicture/result.png')
        rem = "The remedies for Yellow Curl Virus are:\n"
        rem1 =  " 1.Use fungicide such as Benzimidazole fungicides. \n 2.Dig up and dispose affected plant."
        english= "The disease recognised is yellow leaf curl virus and The remedies for Yellow leaf curl virus are Usage of fungicide such as Benzimidazole fungicides and Dig up and dispose affected plant"
        kannada= "kannada"

print(diseasename)
print(rem)
print(rem1)
myobj = gTTS(text=english, lang='en', slow=False)
myobj.save("voice.mp3")
song = MP3('voice.mp3')
print('playing...')
pygame.mixer.init()
pygame.mixer.music.load('voice.mp3')
pygame.mixer.music.play()
time.sleep(song.info.length)
pygame.quit()


