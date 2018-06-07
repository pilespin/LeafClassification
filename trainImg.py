
import tensorflow as tf
import numpy as np
# np.set_printoptions(linewidth=200)

from tfHelper import tfHelper
import data

k = tf.keras

tfHelper.log_level_decrease()

batch_size = 64
num_classes = 10
epochs = 100
imgWidth = 100

print ("Load data ...")
# (x_train, y_train), (x_test, y_test) = data.load_data_train()
(x_train, y_train) = tfHelper.get_dataset_with_folder('classed/', 'L')

# print (x_train)
# print (y_train)

# X_pred, X_id, label = data.load_data_predict()

split = int(len(x_train) * 0.2)
x_test = x_train[:split]
x_train = x_train[split:]
y_test = y_train[:split]
y_train = y_train[split:]

input_size = len(x_train[0])
num_classes = 99

print(str(num_classes) + ' classes')
print(str(input_size) + ' features')
print(str(len(x_train)) + ' lines')

print(x_train.shape, 'train samples')

x_train = data.normalize(x_train)
x_test = data.normalize(x_test)


# model = tfHelper.load_model("model_img")

# model = k.models.Sequential()
# model.add(k.layers.Dense(300, input_dim=input_size, activation='tanh'))
# model.add(k.layers.Dense(200, activation='tanh'))
# model.add(k.layers.Dense(150, activation='tanh'))
# model.add(k.layers.Dense(num_classes, activation='softmax'))

model = k.models.Sequential()
model.add(k.layers.Conv2D(16, (5, 5), activation='relu',
                 input_shape = (imgWidth, imgWidth, 1)))
model.add(k.layers.MaxPool2D(strides=(2,2)))

model.add(k.layers.Flatten())
model.add(k.layers.Dense(1000, activation='relu'))
model.add(k.layers.Dense(num_classes, activation='softmax'))


model.compile(loss='binary_crossentropy'
            , optimizer=k.optimizers.Adam(lr=0.001, decay=0.001)
            , metrics=['accuracy'])

for i in range(3):
	model.fit(x_train, y_train
	            , epochs=1
	            , batch_size=128
            	, shuffle=True
	            , validation_data=(x_test, y_test))
	            # , validation_data=(x_train, y_train))
	
	tfHelper.save_model(model, "model_img")
