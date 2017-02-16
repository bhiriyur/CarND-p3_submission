from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Dense, MaxPooling2D, Dropout
from keras.layers import Flatten, Input, Lambda, ELU, Cropping2D
from keras.preprocessing.image import flip_axis, random_shift
from keras.optimizers import Adam
from keras.regularizers import l2, activity_l2
from keras import backend as K

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm


def read_data(N_VAL=32):
    """
    Reads the driving_log.csv and stores in a pandas dataframe.
    Analyzes the steering info and returns a balanced dataset with
    a smaller balanced subset of frames with all steering angles
    equally represented

    :return: Pandas dataframe with balanced entries

    """
    A = pd.read_csv('data/driving_log.csv')
    n = 11
    bins = np.linspace(-0.6,0.6,n)

    # The default "balanced" spread used to train
    nsample = [17, 70, 210, 525, 832, 832, 525, 210, 70, 17]

    # The spread used to train for sharp turns
    #nsample = [17, 84, 50, 50, 50, 50, 50, 50, 70, 19]

    B = []
    B.append(A[A['steering']<=-0.6])
    B.append(A[A['steering']>  0.6])    
    for i in range(n-1):
        start, end = bins[i], bins[i+1]
        Bi = A[(A['steering']>start) & (A['steering']<=end)]
        B.append(Bi.sample(nsample[i]))
    B = pd.concat(B)

    B = B.sample(frac=1).reset_index(drop=True)
    A_train = B
    A_val = A.sample(N_VAL).reset_index(drop=True)

    return A_train, A_val

def val_data(A):
    """Returns the validation data in a 4th order tensor"""
    N = len(A)
    x,y = [], []
    for i in A.index:
        path = os.path.join('data/', A.center[i].strip())
        xi = image_crop_reshape(plt.imread(path))
        yi = A.steering[i]
        x.append(xi)
        y.append(A.steering[i])

    xval = np.array(x)
    yval = np.array(y)
    print('VALIDATION: {} {}'.format(xval.shape,yval.shape))
    return xval, yval


def get_image_data(A,i,mode=1,flip=0,wshift=0.0,hshift=0.0):
    """Returns the i'th image data from A after applying transformations"""
    lroffset = 0.2
    wshift_factor = 0.5
    
    modes = {1:('center',0.0),
             2:('left',  lroffset),
             3:('right',-lroffset)}
    camera = modes[mode][0]
    shift =  modes[mode][1]

    # Image file path
    path = os.path.join('data',A[camera][i].strip())

    # Load appropriate image and steering with offset
    xi = plt.imread(path)
    yi = A.steering[i]+shift

    # Apply image crop and reshape transformation
    xi = image_crop_reshape(xi)

    # Add translation width and height
    xi = random_shift(xi,wshift,hshift,0,1,2) # image = width x height x ch
    yi += wshift*wshift_factor

    # Flip left/right
    if flip:
        xi = cv2.flip(xi,1)
        yi = -yi

    # TODO: Random variation of darkness

    # TODO: Random addition of shadows

    return xi,yi


def image_crop_reshape(img):
    """Returns a cropped and reshaped image"""
    return cv2.resize(img[70:140, :, :],(200, 200))


def data_generator(A,BATCH_SIZE):
    """ An image data generator"""    
    i = 0
    flip = True
    while True:
        x, y = [], []
        count = 0
        while count < BATCH_SIZE:

            # Pick center (prob = 3/5), left (1/5) or right (1/5) image
            mode = np.random.choice([1, 1, 1, 2, 3], 1)[0]
            #flip = np.random.random()<0.5

            # Random shift in width and height
            wshift,hshift = 0.2*np.random.random(2)-0.1
            xi,yi = get_image_data(A,i,mode,flip,wshift,0.0*hshift)
            x.append(xi)
            y.append(yi)

            # Increment counter for batch
            count += 1

            # Reset to beginning once we reach end
            i += 1
            if i == len(A):
                A = A.sample(frac=1).reset_index(drop=True)
                flip = False
                i = 0

        yield np.array(x), np.array(y)


def nvidia():
    """
    A variant of the nvidia model
    """
    model = Sequential()

    # Takes image after crop and reshape
    img_shape = (200, 200, 3)

    # Normalization
    model.add(Lambda(lambda x: x/127.5 - 1.0,
                     input_shape=img_shape,
                     output_shape=img_shape))
    print("LAYER: {:30s} {}".format('Normalization',model.layers[-1].output_shape))

    # Layer 1
    model.add(Convolution2D(24,3,3,border_mode='same',activation='elu',subsample=(2,2)))
    print("LAYER: {:30s} {}".format('Conv2D-24-3x3-s2',model.layers[-1].output_shape))
    model.add(MaxPooling2D())
    print("LAYER: {:30s} {}".format('Maxpool2D',model.layers[-1].output_shape))

    # Layer 2
    model.add(Convolution2D(36,3,3,border_mode='same',activation='elu',subsample=(2,2)))
    print("LAYER: {:30s} {}".format('Conv2D-36-3x3-s2',model.layers[-1].output_shape))
    model.add(MaxPooling2D())
    print("LAYER: {:30s} {}".format('Maxpool2D',model.layers[-1].output_shape))

    # Layer 3
    model.add(Convolution2D(48,3,3,border_mode='same',activation='elu',subsample=(1,1)))
    print("LAYER: {:30s} {}".format('Conv2D-48-3x3-s1',model.layers[-1].output_shape))
    print("LAYER: {:30s} {}".format('Maxpool2D',model.layers[-1].output_shape))

    # Layer 4
    model.add(Convolution2D(64,3,3,activation='elu',subsample=(1,1)))
    print("LAYER: {:30s} {}".format('Conv2D-64-3x3-s1',model.layers[-1].output_shape))

    # Layer 5
    model.add(Flatten())
    model.add(Dropout(0.5))
    print("LAYER: {:30s} {}".format('Flatten',model.layers[-1].output_shape))

    # Layer 6
    model.add(Dense(500,activation='elu')) #W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Dropout(0.5))
    print("LAYER: {:30s} {}".format('FC',model.layers[-1].output_shape))

    # Layer 7
    model.add(Dense(200,activation='elu'))
    model.add(Dropout(0.5))
    print("LAYER: {:30s} {}".format('FC',model.layers[-1].output_shape))

    # Layer 8
    model.add(Dense(30,activation='elu'))
    print("LAYER: {:30s} {}".format('FC',model.layers[-1].output_shape))
    # Output
    model.add(Dense(1, activation='linear'))
    print("LAYER: {:30s} {}".format('OUTPUT',model.layers[-1].output_shape))

    # Minimization
    adamopt = Adam(lr=0.0001)
    model.compile(loss='mse',optimizer=adamopt)
    return model


def train(FILE='model.h5',load_file=None):
    """
    Build and train the network
    """
    if load_file == None:
        # Generate a new model
        net = nvidia()
    else:
        # For retraining
        net = load_model(load_file)

    for i in range(NB_LOOPS):
        A_train,A_val = read_data()
        xval,yval = val_data(A_val)

        print("Number of examples available = {}".format(A_train.shape[0]))
        print("Batch size = {}".format(BATCH_SIZE))
        print("Samples per epoch = {}".format(N_SAMPLE))


        T = data_generator(A_train,BATCH_SIZE)
        net.fit_generator(T, samples_per_epoch=N_SAMPLE, nb_epoch=NB_EPOCHS,
                          validation_data=(xval,yval), nb_val_samples=N_VAL)

        evaluate(net)
        
    net.save(FILE)
    K.clear_session()
    return net


def evaluate(net):
    #net = load_model(FILE)
    A_train,A_val = read_data()
    y_act, y_pred = [], []
    for i in range(len(A_val)):
        xi,yi = get_image_data(A_val,i,1)
        output = net.predict_on_batch(np.array([xi]))
        print("ACTUAL/PREDICTED  = {:8.4f} {:8.4f}".format(yi,output[0][0]))
        y_act.append(yi)
        y_pred.append(output[0][0])

    #K.clear_session()

    plt.figure()
    plt.plot(y_act,label='Actual',color='b')
    plt.plot(y_pred,label='Predicted',color='r')
    plt.legend(loc='best')
    plt.show()


if __name__=='__main__':

    # Parse input arguments
    import argparse
    prs = argparse.ArgumentParser(description='Input Argument Parser')
    prs.add_argument('-bs','--batch_size',type=int, default = 256, help="Batch size of tensor")
    prs.add_argument('-nl','--num_loops',type=int, default = 1, help="Number of loops")
    prs.add_argument('-ne','--num_epochs',type=int, default = 1, help="Number of loops")    
    prs.add_argument('-ns','--num_samples',type=int, default = 10, help="Number of samples (x batch_size)")
    prs.add_argument('-nv','--num_validation',type=int, default = 32, help="Number of validation samples")
    prs.add_argument('-dt','--drop_threshold',type=float, default = 0.0, help="Dropping threshold for 0 steering")
    prs.add_argument('-tt','--turn_threshold',type=float, default = 0.05, help="Turning angle threshold for 0 steering")
    prs.add_argument('-re','--retrain',type=str, default='', help = "retrain from existing model")
    prs.add_argument('-sf','--save_file',type=str, default='model.h5', help = "File to save model to")
    
    # Parse arguments
    args = prs.parse_args()
    TURN_THRESHOLD = args.turn_threshold
    DROP_THRESHOLD = args.drop_threshold
    N_VAL = args.num_validation
    BATCH_SIZE = args.batch_size
    N_SAMPLE = BATCH_SIZE*args.num_samples
    NB_LOOPS = args.num_loops
    NB_EPOCHS = args.num_epochs
    SAVE_FILE = args.save_file

    print(args)
    
    if args.retrain == '':
        net = train(SAVE_FILE)
    else:
        net = train(SAVE_FILE,args.retrain)

