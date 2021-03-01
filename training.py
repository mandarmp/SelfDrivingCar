import numpy as np
from convNet import build_model
import matplotlib.pyplot as plt
from keras.models import load_model

n_rows = 128
n_cols = 128
alpha = 1e-4
n_epochs = 10

model_name = 'self-driving-car-sim-3d-{}-{}-{}-epochs.model'.format(alpha, 'my cnn', n_epochs)

model = build_model(n_rows, n_cols, alpha)

training_data = np.load('training_data_final.npy')
train_data = training_data[:3200]
test_data = training_data[3200:]

x_train = np.array([data[0] for data in train_data]).reshape(-1, n_rows, n_cols, 1)
y_train = [data[1] for data in train_data]

x_test = np.array([data[0] for data in test_data]).reshape(-1, n_rows, n_cols, 1)
y_test = [data[1] for data in test_data]

model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
hist = model.fit(x_train, np.array(y_train), batch_size=32, epochs=n_epochs,verbose=1,validation_data=(x_test,np.array(y_test)))
model.save("best_model.h5")

#del hist

trained_model = load_model("best_model.h5")
trained_model.summary()



## viewing losses and accuracy
train_loss = hist.history['loss']
train_acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']
xc = range(n_epochs)

print(train_loss)

print(train_acc)

print(val_loss)

print(val_acc)

print(xc)


## first we visualize train loss and validation loss
plt.figure(1,figsize=(7,5))
plt.plot(xc,np.array(train_loss))
plt.plot(xc,np.array(val_loss))
plt.xlabel('no of epochs')
plt.ylabel('loss')
plt.title('train_loss vs validation_loss')
plt.grid(True)
plt.legend(['train','val'])
#
### we visualize train accuracy and validation accuracy
plt.figure(2,figsize=(7,7))
plt.plot(xc,np.array(train_acc))
plt.plot(xc,np.array(val_acc))
plt.xlabel('no of epochs')
plt.ylabel('loss')
plt.title('train acc vs val acc')
plt.grid(True)
plt.legend(['train','val'])
#
#
### we need to evaluate our model for accuracy 
score = model.evaluate(x_test,np.array(y_test),verbose=0)
print('test loss:{}'.format(score[0]))
print('test accuracy:{}'.format(score[1]))
plt.plot(xc,val_acc)


