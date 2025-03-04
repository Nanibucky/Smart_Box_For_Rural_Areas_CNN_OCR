# Importing the required libraries
import cv2                 
import numpy as np         
import os                  
from random import shuffle 
from tqdm import tqdm


# Setting up the environment
TRAIN_DIR = 'C:\\Users\\Thrinu\\Desktop\\New Project\\Plant_Disease_Detection\\train'
TEST_DIR =  'C:\\Users\\Thrinu\\Desktop\\New Project\\Plant_Disease_Detection\\test'

IMG_SIZE = 50
LR = 1e-3
# Setting up the model which will help with tensorflow models
MODEL_NAME = 'plantdiseasedetection-{}-{}.model'.format(LR, '2conv-basic')

# Labelling the dataset
def label_img(img):
    word_label = img[0]
    print(word_label)
  
    if word_label == 'h':
        print('healthy')
        return [1,0,0,0,0,0,0,0]
    elif word_label == 'b':
        print('bacterial')
        return [0,1,0,0,0,0,0,0]
    elif word_label == 'e':
        print('earlyblight')
        return [0,0,1,0,0,0,0,0]
    elif word_label == 'l':
        print('lateblight')
        return [0,0,0,1,0,0,0,0]
    elif word_label == 'm':
        print('leafmold')
        return [0,0,0,0,1,0,0,0]
    elif word_label == 's':
        print('septoria')
        return [0,0,0,0,0,1,0,0]
    elif word_label == 'v':
        print('yellowcurlvirus')
        return [0,0,0,0,0,0,1,0]
    elif word_label == 'z':
        print('negative')
        return [0,0,0,0,0,0,0,1]

# Creating the training data
def create_train_data():
    # Creating an empty list where we should store the training data
    training_data = []

    # tqdm is only used for interactive loading
    # loading the training data
    for img in tqdm(os.listdir(TRAIN_DIR)):
        # labeling the images
        label = label_img(img)
        print('##############')
        print(label)
        path = os.path.join(TRAIN_DIR,img)

        # loading the image from the path 
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        # resizing the image for processing them in the covnet
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        # final step-forming the training data list with numpy array of the images

        training_data.append([np.array(img),np.array(label)])
    # shuffling of the training data to preserve the random state of our data
    shuffle(training_data)

    # saving our trained data for further uses if required
    np.save('train_data.npy', training_data)
    return training_data

# Processing the given test data
# Almost same as processing the training data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
# Running the training and the testing in the dataset for our model
train_data = create_train_data()
# If you have already created the dataset:
#train_data = np.load('train_data.npy')
process_data=process_test_data()

# Creating the neural network using tensorflow
# Importing the required libraries
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf


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
    
# Splitting the testing data and training data
train = train_data[:-4840]
test = train_data[-4840:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]
print(X.shape)
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]
print(test_x.shape)

model.fit({'input': X}, {'targets': Y},n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)











        
