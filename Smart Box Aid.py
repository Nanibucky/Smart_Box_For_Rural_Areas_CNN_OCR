import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
import cv2
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
##import RPi.GPIO as GPIO
import pytesseract
import numpy as np
import os
from random import shuffle
from tqdm import \
    tqdm

##GPIO.setwarnings(False)
##GPIO.setmode(GPIO.BCM)
##
##SW1 = 5
##SW2 = 6
##SW4 = 27
##SW3 = 22
##GPIO.setup(SW1, GPIO.IN)
##GPIO.setup(SW2, GPIO.IN)
##GPIO.setup(SW4, GPIO.IN)
##GPIO.setup(SW3, GPIO.IN)

##GPIO.setwarnings(False)
##GPIO.setmode(GPIO.BCM)

print('Press  Button-1 for Tablet Prescription')
print('Press  Button-2 for Plant Disease Detection')
print('Press  "a" in keyboard for Tablet Prescription')
print('Press  "b" in keyboard for Plant Disease Detection')

while True:
    print('Ta')
##    if event.key == pygame.K_b:
##    if GPIO.input(SW1) == 0:
    if cv2.waitKey(1) & 0xFF == ord('a'):
            print('Tablet Prescription')
            print('Press  Button-1 to Capture Image')
            print('Press "c" to Capture Image')
            vs = cv2.VideoCapture(0)
            while True:
                    (grabbed, frame) = vs.read()
                    if not grabbed:
                            break
                    cv2.imshow('Medicine Tablet', frame)
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                    #if cv2.waitKey(1) & GPIO.input(SW1) == 0:
                            cv2.imwrite('frame.png', frame)
                            break
            vs.release()
            cv2.destroyAllWindows()
            #print('Identifying the Tablet')
            image = "frame.png"
            img = cv2.imread("frame.png")

            # Convert the image to gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Performing OTSU threshold
            ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

            # Specify structure shape and kernel size.
            # Kernel size increases or decreases the area
            # of the rectangle to be detected.
            # A smaller value like (10, 10) will detect
            # each word instead of a sentence.
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

            # Applying dilation on the threshold image
            dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

            # Finding contours
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Creating a copy of image
            im2 = img.copy()
                    
            # Looping through the identified contours
            # Then rectangular part is cropped and passed on
            # to pytesseract for extracting text from it
            # Extracted text is then written into the text file
            text1 =''
            for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Drawing a rectangle on copied image
                    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Cropping the text block for giving input to OCR
                    cropped = im2[y:y + h, x:x + w]
                    
                    # Apply OCR on the cropped image
                    text1 += ' '+pytesseract.image_to_string(cropped)

            text1=text1.lower()      
            print('Recognised Text: ',text1)
            if text1.find('cheston') != -1:
                    english = 'Recognised tablet is Cheston Cold, It is used for Cough and Cold'
                    kannada = 'kannada'

            elif text1.find('saridon') != -1:
                    english = 'Recognised tablet is Saridon, It is used for Headache'
                    kannada = 'kannada'

            elif text1.find('azithromycin') != -1:
                    english = 'Recognised tablet is azithromycin, It is used for Stomach Infection'
                    kannada = 'kannada'

            elif text1.find('lisinopril') != -1:
                    english = 'Recognised tablet is Lisinopril, It is used to control Blood Pressure'
                    kannada = 'kannada'

            elif text1.find('paracetamol') != -1:
                    english = 'Recognised tablet is Paracetamol, It is used for fever and Body Pains'
                    kannada = 'kannada'
            else:
                    english = 'Tablet Not Recognized'
                    kannada = 'kannada'

            print('Recognised Tablet: ',english)
            myobj = gTTS(text=english, lang='en', slow =False)
            myobj.save("voice.mp3")
            song = MP3("voice.mp3")
            pygame.mixer.init()
            pygame.mixer.music.load('voice.mp3')
            pygame.mixer.music.play()
            time.sleep(song.info.length)
            pygame.quit()

##    if GPIO.input(SW2) == 0:
    if cv2.waitKey(1) & 0xFF == ord('b'):
##    if event.key == pygame.K_b:
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
                english= "The disease recognised is Late Blight and The remedies for late blight are Discard any affected leaf or the complete plant if necessary and use antibiotics such as streptomycin or oxytetracycline."    
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
        print('playing....')
        pygame.mixer.init()
        pygame.mixer.music.load('voice.mp3')
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()


        





