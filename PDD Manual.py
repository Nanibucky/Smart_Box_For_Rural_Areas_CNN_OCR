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

window = tk.Tk()

window.title("Plant Disease Detection")

window.geometry("600x600")
window.configure(background ="white")

title = tk.Label(text="Click Select Image button to choose an image from test directory", background = "white", fg="Brown", font=("", 15))
title.grid()

def bacterial():
    
    rem = "The remedies for Bacterial Spot are: "
    remedies = tk.Label(text=rem, background="white",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Discard or destroy any affected plants. \n  Do not compost them. \n  Rotate yoour tomato plants yearly to prevent re-infection next year. \n Use copper fungicites"
    remedies1 = tk.Label(text=rem1, background="white",
                        fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    answer = "Bacterial"
    myobj = gTTS(text=answer, lang='en', slow=False)
    myobj.save("voice.mp3")
    song = MP3('voice.mp3')
    print('playing....')
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()

def earlyblight():

    answer = "Early Blight"
    myobj = gTTS(text=answer, lang='en', slow=False)
    myobj.save("voice.mp3")
    song = MP3('voice.mp3')
    print('playing....')
    
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()

    rem = "The remedies for Early Blight are: "
    remedies = tk.Label(text=rem, background="white",
                          fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Monitor the field, handpick diseased plants and bury them. \n  Use sticky yellow plastic traps. \n  Spray insecticides such as organophosphates, carbametes during the seedliing stage. \n Use copper fungicites"
    remedies1 = tk.Label(text=rem1, background="white",
                             fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

def lateblight():

    answer = "Late Blight"
    myobj = gTTS(text=answer, lang='en', slow=False)
    myobj.save("voice.mp3")
    song = MP3('voice.mp3')
    print('playing....')
    
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()

    rem = "The remedies for Late Blight are: "
    remedies = tk.Label(text=rem, background="white",
                          fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Monitor the field, handpick diseased plants and bury them. \n  Use sticky yellow plastic traps. \n  Spray insecticides such as organophosphates, carbametes during the seedliing stage. \n Use copper fungicites"
    remedies1 = tk.Label(text=rem1, background="white",
                             fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

def leafmold():

    answer = "Leaf Mold"
    myobj = gTTS(text=answer, lang='en', slow=False)
    myobj.save("voice.mp3")
    song = MP3('voice.mp3')
    print('playing....')
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()
    
    rem = "The remedies for Leaf Mold are: "
    remedies = tk.Label(text=rem, background="white",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Monitor the field, handpick diseased plants and bury them. \n  Use sticky yellow plastic traps. \n  Spray insecticides such as organophosphates, carbametes during the seedliing stage. \n Use copper fungicites"
    remedies1 = tk.Label(text=rem1, background="white",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

def septoria():
 
    answer = "Septoria"
    myobj = gTTS(text=answer, lang='en', slow=False)
    myobj.save("voice.mp3")
    song = MP3('voice.mp3')
    print('playing....')
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()

    rem = "The remedies for Septoria are: "
    remedies = tk.Label(text=rem, background="white",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Monitor the field, handpick diseased plants and bury them. \n  Use sticky yellow plastic traps. \n  Spray insecticides such as organophosphates, carbametes during the seedliing stage. \n Use copper fungicites"
    remedies1 = tk.Label(text=rem1, background="white",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

def yellowcurlvirus():

    answer = "Yellow Curl Virus"
    myobj = gTTS(text=answer, lang='en', slow=False)
    myobj.save("voice.mp3")
    song = MP3('voice.mp3')
    print('playing....')
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()

    rem = "The remedies for Yellow Curl Virus are: "
    remedies = tk.Label(text=rem, background="white",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " Monitor the field, remove and destroy infected leaves. \n  Treat organically with copper spray. \n  Use chemical fungicides,the best of which for tomatoes is chlorothalonil."
    remedies1 = tk.Label(text=rem1, background="white",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

def analysis():
    import cv2 
    import numpy as np  
    import os  
    from random import shuffle  
    from tqdm import \
        tqdm  
    
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

    convnet = fully_connected(convnet, 8, activation='softmax')
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
        elif np.argmax(model_out) == 7:
            str_label = 'negative'


        if str_label == 'healthy':
            status= 'Healthy'

            myobj = gTTS(text=status, lang='en', slow=False)
            myobj.save("voice.mp3")
            song = MP3('voice.mp3')
            print('playing....')
            pygame.mixer.init()
            pygame.mixer.music.load('voice.mp3')
            pygame.mixer.music.play()
            time.sleep(song.info.length)
            pygame.quit()
            message = tk.Label(text='Status: '+status, background="white",
                           fg="green", font=("", 15))
            message.grid(column=0, row=4, padx=10, pady=10)

        elif str_label == 'bacterial':
            diseasename = "Bacterial"
            disease = tk.Label(text='Disease Name: ' + diseasename, background="white",
                               fg="green", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)

            bacterial ()
 
        elif str_label == 'earlyblight':
            diseasename = "Early Blight"

            disease = tk.Label(text='Disease Name: ' + diseasename, background="white",
                               fg="green", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)

            earlyblight ()

        elif str_label == 'lateblight':
            diseasename = "Late Blight"

            disease = tk.Label(text='Disease Name: ' + diseasename, background="white",
                               fg="green", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)

            lateblight ()

        elif str_label == 'leafmold':
            diseasename = "leafmold"

            disease = tk.Label(text='Disease Name: ' + diseasename, background="white",
                               fg="green", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)

            leafmold ()

            
        elif str_label == 'septoria':
            diseasename = "Septoria"
            
            disease = tk.Label(text='Disease Name: ' + diseasename, background="white",
                               fg="green", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)

            septoria ()


        elif str_label == 'yellowcurlvirus':
            diseasename = "Yellow Curl Virus"

            disease = tk.Label(text='Disease Name: ' + diseasename, background="white",
                               fg="green", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)

            yellowcurlvirus ()

        elif str_label == 'negative':
            status= 'No Leaf Detected'

            myobj = gTTS(text=status, lang='en', slow=False)
            myobj.save("voice.mp3")
            song = MP3('voice.mp3')
            print('playing....')
            pygame.mixer.init()
            pygame.mixer.music.load('voice.mp3')
            pygame.mixer.music.play()
            time.sleep(song.info.length)
            pygame.quit()
            message = tk.Label(text='Status: '+status, background="white",
                           fg="green", font=("", 15))
            message.grid(column=0, row=4, padx=10, pady=10)

       
def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='C:\\Users\\Thrinu\\Desktop\\New Project\\Plant_Disease_Detection\\test images', title='Select image for analysis ',
                           filetypes=[('image files', '.JPG')])
    dst = "testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    if os.path.split(fileName)[-1].split('.') == 'h (1)':
        print('dfdffffffffffffff')
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="260", width="575")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1 = tk.Button(text="Select Image", command = openphoto)
    button1.grid(column=0, row=3, padx=10, pady = 10)
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady = 10)

button1 = tk.Button(text="Select Image", command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)

window.mainloop()



