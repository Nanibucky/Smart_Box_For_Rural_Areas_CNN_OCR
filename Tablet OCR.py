import shutil
import os
import sys
from PIL import Image, ImageTk
import cv2
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import pytesseract
import os

while True:
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
                kannada = 'Gurutisalpaṭṭa ṭyābleṭ cesṭan śītavāgide, idannu kemmu mattu śītakke baḷasalāguttade'

        elif text1.find('saridon') != -1:
                english = 'Recognised tablet is Saridon, It is used for Headache'
                kannada = 'Gurutisalpaṭṭa ṭyābleṭ cesṭan śītavāgide, idannu kemmu mattu śītakke baḷasalāguttade'

        elif text1.find('azithromycin') != -1:
                english = 'Recognised tablet is azithromycin, It is used for Stomach Infection'
                kannada = 'Gurutisalpaṭṭa ṭyābleṭ cesṭan śītavāgide, idannu kemmu mattu śītakke baḷasalāguttade'

        elif text1.find('lisinopril') != -1:
                english = 'Recognised tablet is Lisinopril, It is used to control Blood Pressure'
                kannada = 'Gurutisalpaṭṭa ṭyābleṭ cesṭan śītavāgide, idannu kemmu mattu śītakke baḷasalāguttade'

        elif text1.find('paracetamol') != -1:
                english = 'Recognised tablet is Paracetamol, It is used for fever and Body Pains'
                kannada = 'Gurutisalpaṭṭa ṭyābleṭ cesṭan śītavāgide, idannu kemmu mattu śītakke baḷasalāguttade'
        else:
                english = 'Tablet Not Recognized'
                kannada = 'Ṭyābleṭ gurutisalāgilla'

        print('Recognised Tablet: ',english)
        myobj = gTTS(text=kannada, lang='kn', slow =False)
        myobj.save("voice.mp3")
        song = MP3("voice.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load('voice.mp3')
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()
