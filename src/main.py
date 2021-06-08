from model.model import create_model
from dataset.load_dataset import fetch_dataset


(x_train,y_train),(x_test, y_test) = fetch_dataset()

# creating the model
model = create_model()

# training the model
batch_size = 128
epochs = 10

# taining the model
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

# saving the model
model.save('./model/mnist.h5')

# testing model
accuracy = model.evaluate(x_test,y_test,verbose=0)

print('Test Loss     :', accuracy[0])
print('Test accuracy :',accuracy[1])


