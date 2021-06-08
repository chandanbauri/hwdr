from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import Conv2D , MaxPooling2D, Flatten,Dropout , Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta

def create_model():

    input_shape = (28,28,1)
    num_classes = 10

    model = Sequential()
    model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='relu'))

    model.compile(loss=categorical_crossentropy,optimizer=Adadelta(),metrics=['accuracy'])

    return model