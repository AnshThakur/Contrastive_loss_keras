import re
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

#################################################### ENERGY function
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

########################## Contrastive Loss

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

######################################################

size = 22 ## to downsample images
total_sample_size = 10000

############################################################# returns numpy array by reading image

def read_image(filename, byteorder='>'):
    #first we read the image, as a raw file to the buffer
    with open(filename, 'rb') as f:
        buffer = f.read()
    #using regex, we extract the header, width, height and maxval of the image
    header, width, height, maxval = re.search(
        b"(^P5\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    #then we convert the image to numpy array using np.frombuffer which interprets buffer as one dimensional array
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))



#############################################################################

def get_data(size, total_sample_size):
    #read the image
    image = read_image('orl_faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    #reduce the size
    image = image[::size, ::size]
    #get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]
    
    count = 0
    
    #initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])  # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])
    
    for i in range(40):
        for j in range(int(total_sample_size/40)):
            ind1 = 0
            ind2 = 0
            
            #read images from same directory (genuine pair)
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)
            
            # read the two images
            img1 = read_image('orl_faces/s' + str(i+1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            img2 = read_image('orl_faces/s' + str(i+1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')
            
            #reduce the size
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
            

            #store the images to the initialized numpy array
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2
            
            #as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])
    
    for i in range(int(total_sample_size/10)):
        for j in range(10):
            
            #read images from different directory (imposite pair)
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break
                    
            img1 = read_image('orl_faces/s' + str(ind1+1) + '/' + str(j + 1) + '.pgm', 'rw+')
            img2 = read_image('orl_faces/s' + str(ind2+1) + '/' + str(j + 1) + '.pgm', 'rw+')

            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            #as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_imposite[count] = 0
            count += 1
            
    #now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)/255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y


#############################################################

X, Y = get_data(size, total_sample_size)

print(X.shape)
print(Y.shape)

#############################
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)

################################### BASE NETWORK

def build_base_network():
    inputs=Input(shape=(6,5,1), name='in_layer') 
    #convolutional layer 1
    conv1=Conv2D(6,(3, 3),padding="same", activation="relu")(inputs)
    maxp1=MaxPooling2D(pool_size=(2, 2))(conv1)    
    conv2=Conv2D(12,(3, 3),padding="same", activation="relu")(maxp1)
    maxp2=MaxPooling2D(pool_size=(2, 2))(conv2)  
    naxp2=Dropout(0.25)(maxp2) 
    F=Flatten()(maxp2)
    d1=Dense(128, activation='relu')(F)
    d1=Dropout(0.1)(d1)
    d2=Dense(50, activation='relu')(d1)
    model=Model(inputs,d2)
    return model

############################################################### CREATE SIAMESE NET

img_a = Input(shape=(6,5,1))
img_b = Input(shape=(6,5,1))
base_network = build_base_network()
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
model = Model(input=[img_a, img_b], output=distance)

epochs = 100


model.compile(loss=contrastive_loss, optimizer='nadam') ### NADAM optimizer
model.summary()


################################

# Create data for training

img_1 = x_train[:, 0]
img2 = x_train[:, 1]
img_1=np.rollaxis(img_1, 2, 1)
img_1=np.rollaxis(img_1, 3, 2)
print('final')
print(img_1.shape)
img2=np.rollaxis(img2, 2, 1)
img2=np.rollaxis(img2, 3, 2)
print('final')
print(img2.shape)
model.fit([img_1, img2], y_train, validation_split=.25,
          batch_size=128, verbose=2, nb_epoch=epochs)

#########################################################
test=x_test[:, 0]
test=np.rollaxis(test, 2, 1)
test=np.rollaxis(test, 3, 2)

test1=x_test[:, 1]
test1=np.rollaxis(test1, 2, 1)
test1=np.rollaxis(test1, 3, 2)



pred = model.predict([test,test1])

###############################################
def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

############################################

print(compute_accuracy(pred, y_test))

















