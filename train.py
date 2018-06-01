
import tensorflow as tf
import numpy as np
# np.set_printoptions(linewidth=200)

from tfHelper import tfHelper

import data

k = tf.keras

tfHelper.log_level_decrease()
# tfHelper.numpy_show_entire_array(28)
# np.set_printoptions(threshold='nan', linewidth=114)
# np.set_printoptions(linewidth=114)

batch_size = 64
num_classes = 10
epochs = 100
imgWidth = 28

print ("Load data ...")
(x_train, y_train), (x_test, y_test) = data.load_data_train()
X_pred, X_id, label = data.load_data_predict()

input_size = len(x_train[0])

num_classes = 99
print(str(num_classes) + ' classes')
print(str(input_size) + ' features')
print(str(len(x_train)) + ' lines')

# print (y_train)
# exit(0)

# model = tfHelper.load_model("model")

model = k.models.Sequential()
model.add(k.layers.Dense(300, input_dim=input_size, activation='tanh'))
model.add(k.layers.Dense(200, activation='tanh'))
model.add(k.layers.Dense(150, activation='tanh'))
model.add(k.layers.Dense(num_classes, activation='softmax'))


model.compile(loss='binary_crossentropy'
            , optimizer=k.optimizers.Adam(lr=0.01, decay=0.001)
            , metrics=['accuracy'])

# for i in range(1000):
model.fit(x_train, y_train
            , epochs=100
            , batch_size=32
            , validation_data=(x_test, y_test))
            # , validation_data=(x_train, y_train))

tfHelper.save_model(model, "model")


# model = k.models.Sequential()
# model.add(k.layers.Conv2D(16, (3, 3), activation='relu',
#                  input_shape = (imgWidth, imgWidth, 1)))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.MaxPool2D(strides=(2,2)))
# model.add(k.layers.Dropout(0.25))

# model.add(k.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.MaxPool2D(strides=(2,2)))
# model.add(k.layers.Dropout(0.25))

# model.add(k.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.MaxPool2D(strides=(2,2)))
# model.add(k.layers.Dropout(0.25))

# model.add(k.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.MaxPool2D(strides=(2,2)))
# model.add(k.layers.Dropout(0.25))

# model.add(k.layers.Flatten())
# model.add(k.layers.Dense(1024, activation='relu'))
# model.add(k.layers.Dropout(0.2))
# model.add(k.layers.Dense(1024, activation='relu'))
# model.add(k.layers.Dropout(0.2))
# model.add(k.layers.Dense(100, activation='relu'))
# model.add(k.layers.Dropout(0.2))
# model.add(k.layers.Dense(num_classes, activation='softmax'))


# opt = k.optimizers.Adam(lr=1e-4)
# # opt = k.optimizers.Adam(lr=0.0001, decay=1e-6)
# # opt = k.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# learning_rate_reduction = k.callbacks.ReduceLROnPlateau(monitor='val_loss', 
#                                                         patience=1, 
#                                                         verbose=1, 
#                                                         factor=0.5, 
#                                                         min_lr=1e-09)

# tensorBoard = k.callbacks.TensorBoard()

# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

# datagen = k.preprocessing.image.ImageDataGenerator( rotation_range=20,
#                                                     width_shift_range=0.1,
#                                                     height_shift_range=0.1,
#                                                     # shear_range=0.2,
#                                                     zoom_range=0.1,
#                                                     # horizontal_flip=True,
#                                                     fill_mode='nearest')
# datagen.fit(x_train)

# # model.fit(x_train, y_train,
# for i in range(epochs):
#     print("Epoch " + str(i) + '/' + str(epochs))
#     model.fit_generator(datagen.flow(x_train, y_train,
#             batch_size=batch_size),
#             epochs=1,
#             # validation_data=(x_train, y_train),
#             # steps_per_epoch=10,
#             validation_data=(x_test, y_test),
#             # shuffle=True,
#             verbose=1,
#             callbacks=[learning_rate_reduction, tensorBoard]
#             )

#     tfHelper.save_model(model, "model")
