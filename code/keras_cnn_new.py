#from __future__ import print_function
import numpy as np
#import h5py
np.random.seed(1337)  # for reproducibility
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint

batch_size = 64 
nb_classes = 2
nb_epoch = 350

# input image dimensions
img_rows, img_cols = 229,229
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


#model = VGG16(weights='imagenet', include_top=False)
#x = Dense(1, activation='sigmoid', name='predictions')(model.layers[-2].output)
#my_model = Model(input=model.input, output=x)
#my_model.summary()
#model=my_model

model_Xception_conv = Xception(weights='imagenet', include_top=False)
#model_vgg16_conv.summary()

#Create your own input format (here 3x200x200)
input = Input(shape=(229,229,3),name = 'image_input')

#Use the generated model 
output_Xception_conv = model_Xception_conv(input)
for i, layer in enumerate(model_Xception_conv.layers):
    print(layer)
    if i <= 111:
        layer.trainable = False
model_Xception_conv.summary()
#Conv layers
x = ZeroPadding2D((1,1), name='zeropadding2d_43')(output_Xception_conv)
x = Convolution2D(128, 3, 3)(x)#, activation='relu', name='conv7')(x)a
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), strides=(2,2))(x)
#Fully-connected layers 
x = Flatten(name='flatten')(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid', name='predictions')(x)

my_model = Model(input=input, output=x)

#In the summary, weights and layers from Xception part will be hidden, but they will be fit during training
my_model.summary()
model = my_model
#model.load_weights('keras_cnn_new_second_run.h5')
#model= keras.models.load_model('/data/NN/checkpoints/model_checkpoint_keras_cnn_new.hdf5')
model.summary()
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        #shear_range=0.2,
        zoom_range=0.2,
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True)

#train_datagen.fit()
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        '/data/train',
        target_size = (img_rows,img_cols),
        batch_size=batch_size,
        class_mode='binary',
        classes = ['Benign','Malignant'])

validation_generator = test_datagen.flow_from_directory(
        '/data/validation',
        target_size=(img_rows, img_cols),
        batch_size=32,
        class_mode='binary',
        classes=['Benign', 'Malignant'])
checkpointer = ModelCheckpoint(filepath = '/data/NN/checkpoints/model_checkpoint_keras_cnn_new.hdf5',period = 10)
#TODO:put real samples per epoch val in later
history = model.fit_generator(
        train_generator,
        samples_per_epoch=4096, 
        nb_epoch=nb_epoch,
        verbose=1,
        validation_data=validation_generator,
        nb_val_samples=195,
	callbacks = [checkpointer])
#score = model.evaluate(X_test, Y_test, verbose=0)
write_to = open('test_hist_save.txt', 'w')
result = np.zeros([350,4])
#Record loss and accuracy in .npy
for i,items in enumerate(history.history.keys()):
    result[:,i] = history.history[items]
    write_to.write(items)
    write_to.write(': ')
    write_to.write('%s\n' % history.history[items])
np.save('test_hist_numpy.npy',result)
write_to.close()
model.save_weights('keras_cnn_new___current_weights.h5')
