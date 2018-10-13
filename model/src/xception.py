import os
import numpy
import pickle
import keras
import keras.utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.applications import Xception
from keras.models import Sequential
from keras.preprocessing import image
from keras import backend as k
from keras.callbacks import Callback
from tensorflow.contrib.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
from PIL import Image
import png
from matplotlib.pyplot import imshow
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import callbacks
from tensorflow.contrib.keras import regularizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import read_data as rd

img_dir = '../spectrogram_images_299'
img_height = 299
img_width = 299
L2_LAMBDA = 0.001

# hyper parameters for model
nb_classes = 10  # number of classes
based_model_last_block_layer_number = 16  # value is based on based model selected.
BATCH_SIZE = 8  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 30 # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .001  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation

n_samples     = 1000                              # change to 1000 for entire dataset

label_dict = {'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4, 'jazz':5, 'metal':6, 'pop':7, 'reggae':8, 'rock':9}

def xception():
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights

    base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())  # Flatten output and send it to MLP

    # 1-layer MLP with Dropout, BN
    model.add(Dense(256, name='dense_1', kernel_regularizer=regularizers.l2(L2_LAMBDA)))
    model.add(Dropout(rate=0.3, name='dropout_1'))  # Can try varying dropout rates
    model.add(Activation(activation='relu', name='activation_1'))

    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
    base_model.trainable =False
    model.summary()

    loss = 'categorical_crossentropy'

    metrics = ['categorical_accuracy']

    

    model.compile(optimizer='nadam', loss=loss, metrics=metrics)
    return model

def load_batch(file_list):
    img_array = []
    idx_array = []
    label_array = []

    for file_ in file_list:
        
        im = Image.open(img_dir + "/" + file_).convert('RGB')
        
        im = im.resize((img_width, img_height), Image.ANTIALIAS)
        img_array.append(numpy.array(im))

        vals = file_[:-4].split('.')
        idx_array.append(vals[1])
        label_array.append([label_dict[vals[0]]])

    label_array = one_hot.fit_transform(label_array).toarray()
    img_array = numpy.array(img_array)/255.0 # Normalize RGB
    
    return img_array, numpy.array(label_array), numpy.array(idx_array)

def batch_generator(files, BATCH_SIZE):
    L = len(files)

    #this line is just to make the generator infinite, keras needs that    
    while True:

        batch_start = 0
        batch_end = BATCH_SIZE

        while batch_start < L:
            
            limit = min(batch_end, L)
            file_list = files[batch_start: limit]
            batch_img_array, batch_label_array, batch_idx_array = load_batch(file_list)

            yield (batch_img_array, batch_label_array) # a tuple with two numpy arrays with batch_size samples     

            batch_start += BATCH_SIZE   
            batch_end += BATCH_SIZE


def one_hot_encoder(true_labels, num_records, num_classes):
    temp = numpy.array(true_labels[:num_records])
    true_labels = numpy.zeros((num_records, num_classes))
    true_labels[numpy.arange(num_records), temp] = 1
    return true_labels    

    print("Done Training")
    print("AUC now")


if __name__ == '__main__':
    #rd.extract_images()
    # Getting spectrogram images
    
    i = 0
    one_hot = OneHotEncoder(n_values = 10)
    rawdata = []
    

    all_files = os.listdir(img_dir)
    labels_array = []
    
    for file_ in all_files:
        vals = file_[:-4].split('.')
        labels_array.append(label_dict[vals[0]])

    cl_weight = compute_class_weight(class_weight = 'balanced', classes = numpy.unique(labels_array), y = labels_array)
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(all_files, labels_array, random_state = 10, test_size=0.2)
    val_files, X_test, val_labels, Y_test = train_test_split(X_test, Y_test, random_state = 10, test_size = 0.5)
    
    model = xception()
    
    filepath="saved_models/transfer_learning_epoch_{epoch:02d}.h5"
    

    #save model
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', 
                                       verbose=0, 
                                       save_best_only=False)
    callbacks_list = [checkpoint]

    STEPS_PER_EPOCH = len(X_train)//BATCH_SIZE
    VAL_STEPS = len(val_files)//BATCH_SIZE
    # train model

    print("START TRAINING")
    train_loss, train_acc, test_loss, test_acc, train_auc = [], [], [], [], []
    history = model.fit_generator(generator = batch_generator(X_train, BATCH_SIZE), epochs = nb_epoch, steps_per_epoch= STEPS_PER_EPOCH,class_weight = cl_weight, validation_data=batch_generator(val_files, BATCH_SIZE), validation_steps = VAL_STEPS,  callbacks = callbacks_list)
    
    #get accuracy(auc) model  
    model = models.load_model(filepath='saved_models/transfer_learning_epoch_29.h5')

    TEST_STEPS = len(X_test)//BATCH_SIZE

    pred_probs = model.predict_generator(generator = batch_generator(X_test, BATCH_SIZE), steps=TEST_STEPS)

    pred = numpy.argmax(pred_probs, axis=-1)

    one_hot_true = one_hot_encoder(Y_test, len(pred), len(label_dict))

    auc = roc_auc_score(y_true=one_hot_true, y_score=pred_probs, average='macro')

    print('ROC AUC = {0:.3f}'.format(auc))
    prec = average_precision_score(one_hot_true, pred_probs)
    print('Average Precision-Recall Score = {0:.3f}'.format(prec))  
    
