#keras in backend uses the tensorflow

from keras.models import Sequential
from keras.layers import Convolution2D , MaxPooling2D , Dense , Dropout ,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import scipy

 #to preprocessing ur images

train_data_gen=ImageDataGenerator(rescale=1./255)                                             # image genertaor for traning purpose
valid_data_gen=ImageDataGenerator(rescale=1./255)                                             # for testing purpose
 #preprocess all train data


train_generator= train_data_gen.flow_from_directory(
    'data/train',                       # path given from data process train folder with flow from directory we can flow though the data and collect the data
    target_size=(48, 48),                             # the path given then data ma usme train ma jake jo bhi data hoga use collect krlega mtlb images ko hmri sri
    batch_size=64,                                                          # with target size we can size the images into 48 * 48
    color_mode="grayscale",                                   # batch size is thhe number of samples that are passed to the network at once(each training session)
    class_mode='categorical'                                       # class mode specifies the type of label arrays that are returned.

)
# preprocess all test images
valid_generator = valid_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'

)
#  convolutional neural network  : good at detecting pattern and feature in images,videos,audio
# create model structure

emotional_model = Sequential()

emotional_model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))   # activation relu are for hidden layers , input size mai 48*48 pixel diye and ak hi color pass ho rh hota 1 hojuga nd rgb hote toh 3 hote
emotional_model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
emotional_model.add(MaxPooling2D(pool_size= (2, 2)))                                         # pooling layer k liye
emotional_model.add(Dropout(0.25))                                                           # overfitting na ho isliye 0.25 k drop out denge

emotional_model.add(Convolution2D(128, kernel_size=(3, 3) , activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=(2, 2)))
emotional_model.add(Convolution2D(128, kernel_size=(3, 3) , activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=(2, 2)))
emotional_model.add(Dropout(0.25))

emotional_model.add(Flatten())
emotional_model.add(Dense(1024, activation='relu'))
emotional_model.add(Dropout(0.5))
emotional_model.add(Dense(7, activation='softmax'))                  # softmax for output layer  DENSE M 7 KA MTLB 7 EMOTION

# to compile this convolutional network
optimizer = Adam(learning_rate=0.0001)

# Compile the model using the defined optimizer
emotional_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) # loss to use to evaluates set of weights ,optimizer used to search through different weights for the network,metrics you want to collect nd report during traning

#loss function, optimizer , metric funtion used lr=learning   loss funtion used to calculate the difference bwtween the predicted output and the actuaal output   , metric used to judge the performance of model


# train this neural network with preprocess data
#we will fit the data th whole data stored in tarin generator by passing it to fit generator into emotional model info

emotional_model_info= emotional_model.fit_generator(
     train_generator,
     steps_per_epoch=28709//64,                                                # total no of images by 64
     epochs=50,                                                                # set epochs acc to computation as they consume lot of time
     validation_data=valid_generator,                                            # evaluate how well our model working
     validation_steps=7178//64                                # epoch is one pass through all the rows in traning data set



 )   # after whole this our model get trained and evaluated
  # we can store our model structure  noe in json file
model_json = emotional_model.to_json()
with open("emotional_model.json","w") as json_file:
    json_file.write(model_json)


#save trained model weight in .h5 file
emotional_model.save_weights('emotion_model.h5')






