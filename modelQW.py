import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
from keras import initializers
from keras import regularizers


def colearning(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])  # only pads dim 2 and 3 (h and w)

    [ inputtemp, inputspet,inputsct] = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': 3})(inputs)

    conv1ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputsct)
    conv1ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1ct)
    pool1ct = MaxPooling2D(pool_size=(2, 2))(conv1ct)
    conv2ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1ct)
    conv2ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2ct)
    pool2ct = MaxPooling2D(pool_size=(2, 2))(conv2ct)
    conv3ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2ct)
    conv3ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3ct)
    pool3ct = MaxPooling2D(pool_size=(2, 2))(conv3ct)
    conv4ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3ct)
    conv4ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4ct)
    drop4ct = Dropout(0.5)(conv4ct)
    pool4ct = MaxPooling2D(pool_size=(2, 2))(conv4ct)

    conv1pet = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputspet)
    conv1pet = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1pet)
    pool1pet = MaxPooling2D(pool_size=(2, 2))(conv1pet)
    conv2pet = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1pet)
    conv2pet = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2pet)
    pool2pet = MaxPooling2D(pool_size=(2, 2))(conv2pet)
    conv3pet = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2pet)
    conv3pet = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3pet)
    pool3pet = MaxPooling2D(pool_size=(2, 2))(conv3pet)
    conv4pet = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3pet)
    conv4pet = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4pet)
    drop4pet = Dropout(0.5)(conv4pet)
    pool4pet = MaxPooling2D(pool_size=(2, 2))(conv4pet)


    comerge1_temp = concatenate([pool1ct, pool1pet], axis=3)
    poolctexp_temp = Lambda(expand_dim_backend, arguments={'dim': (4)})(pool1ct)
    poolpetexp_temp = Lambda(expand_dim_backend, arguments={'dim': (4)})(pool1pet)

    comerge2_temp = concatenate([poolctexp_temp, poolpetexp_temp], axis=4)
    input_mm = Lambda(tf.pad, arguments={'paddings': (paddings), 'mode': ("CONSTANT")})(comerge2_temp)
    input_mm = Lambda(tf.transpose, arguments={'perm': ([0, 4, 1, 2, 3])})(input_mm)
    comerge2con_temp = Conv3D(filters=128, kernel_size=[2,3,3],
                         kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None),
                         kernel_regularizer=regularizers.l2(0.1),
                         bias_initializer='zeros', padding='valid', activation='relu')(input_mm)
    colearn_out_temp = Lambda(tf.squeeze, arguments={'squeeze_dims': (1)})(comerge2con_temp)
    conj1 = Lambda(tf.multiply, arguments={'y': (comerge1_temp)})(colearn_out_temp)

    comerge1_temp = concatenate([pool2ct, pool2pet], axis=3)
    poolctexp_temp = Lambda(expand_dim_backend, arguments={'dim': (4)})(pool2ct)
    poolpetexp_temp = Lambda(expand_dim_backend, arguments={'dim': (4)})(pool2pet)

    comerge2_temp = concatenate([poolctexp_temp, poolpetexp_temp], axis=4)
    input_mm = Lambda(tf.pad, arguments={'paddings': (paddings), 'mode': ("CONSTANT")})(comerge2_temp)
    input_mm = Lambda(tf.transpose, arguments={'perm': ([0, 4, 1, 2, 3])})(input_mm)

    comerge2con_temp = Conv3D(filters=128, kernel_size=[2,3,3],
                         kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None),
                         kernel_regularizer=regularizers.l2(0.1),
                         bias_initializer='zeros', padding='valid', activation='relu')(input_mm)
    colearn_out_temp = Lambda(tf.squeeze, arguments={'squeeze_dims': (1)})(comerge2con_temp)
    conj2 = Lambda(tf.multiply, arguments={'y': (comerge1_temp)})(colearn_out_temp)


    comerge1_temp = concatenate([pool3ct, pool3pet], axis=3)
    poolctexp_temp = Lambda(expand_dim_backend, arguments={'dim': (4)})(pool3ct)
    poolpetexp_temp = Lambda(expand_dim_backend, arguments={'dim': (4)})(pool3pet)

    comerge2_temp = concatenate([poolctexp_temp, poolpetexp_temp], axis=4)

    input_mm = Lambda(tf.pad, arguments={'paddings': (paddings), 'mode': ("CONSTANT")})(comerge2_temp)
    input_mm = Lambda(tf.transpose, arguments={'perm': ([0, 4, 1, 2, 3])})(input_mm)
    comerge2con_temp = Conv3D(filters=128, kernel_size=[2,3,3],
                         kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None),
                         kernel_regularizer=regularizers.l2(0.1),
                         bias_initializer='zeros', padding='valid', activation='relu')(input_mm)
    colearn_out_temp = Lambda(tf.squeeze, arguments={'squeeze_dims': (1)})(comerge2con_temp)
    conj3 = Lambda(tf.multiply, arguments={'y': (comerge1_temp)})(colearn_out_temp)


    comerge1_temp = concatenate([pool4ct, pool4pet], axis=3)
    poolctexp_temp = Lambda(expand_dim_backend, arguments={'dim': (4)})(pool4ct)
    poolpetexp_temp = Lambda(expand_dim_backend, arguments={'dim': (4)})(pool4pet)

    comerge2_temp = concatenate([poolctexp_temp, poolpetexp_temp], axis=4)
    input_mm = Lambda(tf.pad, arguments={'paddings': (paddings), 'mode': ("CONSTANT")})(comerge2_temp)
    input_mm = Lambda(tf.transpose, arguments={'perm': ([0, 4, 1, 2, 3])})(input_mm)

    comerge2con_temp = Conv3D(filters=128, kernel_size=[2,3,3],
                         kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None),
                         kernel_regularizer=regularizers.l2(0.1),
                         bias_initializer='zeros', padding='valid', activation='relu')(input_mm)
    colearn_out_temp = Lambda(tf.squeeze, arguments={'squeeze_dims': (1)})(comerge2con_temp)
    conj4 = Lambda(tf.multiply, arguments={'y': (comerge1_temp)})(colearn_out_temp)

    up5 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conj4))
    conv5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    merge5 = concatenate([conj3, conv5], axis=3)

    up6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(merge5))
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    merge6 = concatenate([conj2, conv6], axis=3)

    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(merge6))

    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    merge7 = concatenate([conj1, conv7], axis=3)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(merge7))

    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    conv9 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)


    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])



    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def expand_dim_backend(x,dim):
    xe = K.expand_dims(x, dim)
    return xe
