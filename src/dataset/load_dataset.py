from tensorflow.keras.datasets import mnist # mnist dataset
from tensorflow.keras.utils import to_categorical



def fetch_dataset():

    # splitting the dataset
    (x_train,y_train),(x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0],28,28,1)
    x_test = x_test.reshape(x_test.shape[0],28,28,1)

    # converting the output classes into binary matrices
    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)

    # type casting the input sets to float
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')

    # Normalizing the values with in the range 0f 0 - 1 
    x_train /= 255
    x_test /= 255

    return (x_train,y_train),(x_test, y_test)