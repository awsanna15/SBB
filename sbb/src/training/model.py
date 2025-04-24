import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, regularizers


class SBBModeling:
    @staticmethod
    def build_resnet_reg(im_size: int):
        dropout_rate = 0.3
        dense_last_layer = 64  
        l2_reg = 0.001  

        im_input = layers.Input(shape=(im_size, im_size, 1))

        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(im_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        res = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(l2_reg))(x)
        res = layers.BatchNormalization()(res)
        res = layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg))(res)
        res = layers.BatchNormalization()(res)
        x = layers.add([x, res])  # skip connection
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)


        res = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(l2_reg))(x)
        res = layers.BatchNormalization()(res)
        res = layers.Conv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg))(res)
        res = layers.BatchNormalization()(res)
        
        x = layers.Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.add([x, res]) 
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(dense_last_layer, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(2, activation='softmax')(x)  


        model = models.Model(inputs=im_input, outputs=x)


        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        return model
     

    def build_resnet(im_size: int):
        """builds a model with ResNet50 as backbone

        Returns:
            _type_: _description_
        """
        dropout = 0.2
        dense_last_layer = 32 #128
        
        im_input = tf.keras.Input((im_size, im_size, 1))
        
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights=None,
            #pooling='avg',
            input_tensor=im_input
        )
        
        head_model = base_model.output
        # head_model = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(head_model)
        head_model = tf.keras.layers.Flatten(name="flatten")(head_model)
        head_model = tf.keras.layers.Dense(dense_last_layer * 4, activation='sigmoid')(head_model)
        
        head_model = tf.keras.layers.Dense(dense_last_layer, activation="sigmoid")(head_model)
        head_model = tf.keras.layers.Dropout(dropout)(head_model)
        #head_model = tf.keras.layers.Dense(1, activation="sigmoid")(head_model)
        head_model = tf.keras.layers.Dense(2, activation="softmax")(head_model)

        model = tf.keras.models.Model(inputs=base_model.input, outputs=head_model)
        
        optim = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        
        model.compile(
            optimizer=optim,
            #loss=tf.keras.losses.BinaryCrossentropy(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        model.summary()
        return model
    
    @staticmethod
    def build_custom_resnet(im_size: int):
        """_summary_
        builds a custom residual network which is dervied from a unet style piece
        Args:
            im_size (int): _description_

        Returns:
            _type_: _description_
        """
        dense_last_layer = 32
        dropout = 0.2
        
        im_input = tf.keras.Input((im_size, im_size, 1))
        
        #Contraction path
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(im_input)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        b1 = tf.keras.layers.BatchNormalization()(c1)
        r1 = tf.keras.layers.ReLU()(b1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        b2 = tf.keras.layers.BatchNormalization()(c2)
        r2 = tf.keras.layers.ReLU()(b2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)
        
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        b3 = tf.keras.layers.BatchNormalization()(c3)
        r3 = tf.keras.layers.ReLU()(b3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)
        
        head_model = tf.keras.layers.Flatten(name="flatten")(p3)
        head_model = tf.keras.layers.Dense(dense_last_layer * 4, activation='sigmoid')(head_model)
        
        head_model = tf.keras.layers.Dense(dense_last_layer, activation="sigmoid")(head_model)
        head_model = tf.keras.layers.Dropout(dropout)(head_model)
        #head_model = tf.keras.layers.Dense(1, activation="sigmoid")(head_model)
        head_model = tf.keras.layers.Dense(2, activation="softmax")(head_model)
        model = tf.keras.models.Model(inputs=im_input, outputs=head_model)
        model.summary()
        
        optim = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        
        model.compile(
            optimizer=optim,
            #loss=tf.keras.losses.BinaryCrossentropy(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model
    
    @staticmethod
    def build_origin_model(im_size: int):
        """
        """
        num_filters = 32                # Number of conv filters
        max_pool_size = (2, 2)          # shape of MaxPool
        conv_kernel_size = (3, 3)       # conv kernel shape
        imag_shape = (im_size, im_size, 1)
        num_classes = 2
        drop_prob = 0.3                 # fraction to drop (0-1.0)


        model = Sequential()
        # Define layers in the NN
        # Define the 1st convlution layer.  We use border_mode= and input_shape only on first layer
        # border_mode=value restricts convolution to only where the input and the filter fully overlap (ie. not partial overlap)
        #model.add(tf.keras.Input(shape=imag_shape))
        model.add(tf.keras.layers.Conv2D(num_filters, conv_kernel_size))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=max_pool_size))

        # 2nd Convolution layer
        model.add(tf.keras.layers.Conv2D(num_filters * 2, conv_kernel_size))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=max_pool_size))

        # 3rd Convolution layer
        model.add(tf.keras.layers.Conv2D(num_filters * 4, conv_kernel_size))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=max_pool_size))

        # Fully Connected Layer
        model.add(Flatten())
        model.add(Dense(128))   # Fully connected layer in Keras
        model.add(Activation('relu'))

        # dropout some neurons to reduce overfitting
        model.add(Dropout(drop_prob))

        # Readout layer
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # Set loss and measurement, optimizer, and metric used to evaluate loss
        model.compile(
            # loss='sparse_categorical_crossentropy',
            #loss=tf.keras.losses.BinaryCrossentropy(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            # optimizer='adam',   # was adadelta
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            metrics=['accuracy']
        )
        # model.summary()
        return model
    
    @staticmethod
    def build_model(im_size: int):
        """_summary_
        creates the sbb model for bridge classification
        Args:
            im_size (int): size of the image which are fed in
        """
        
        num_filters = 32
        max_pool_size = (2, 2)
        conv_kernel_size = (3, 3)
        im_shape = (im_size, im_size, 1)
        num_classes = 2
        drop_prob = 0.5
        
        im_input = tf.keras.Input(im_shape)
        conv1 = tf.keras.layers.Conv2D(num_filters, conv_kernel_size, padding='same', activation='relu')(im_input)
        pool1 = MaxPooling2D(pool_size=max_pool_size)(conv1)
        
        conv2 = tf.keras.layers.Conv2D(num_filters, conv_kernel_size, padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=max_pool_size)(conv2)
        
        down = tf.keras.layers.Conv2D(num_filters, conv_kernel_size, padding='same', strides=2)(pool1)
        add = tf.keras.layers.add([down, pool2])
        
        x = Flatten(input_shape=(im_size, im_size))(add)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dropout(drop_prob)(x)
        
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        
        model = tf.keras.models.Model(inputs=im_input, outputs=x)
        model.summary()
        # exit(0)
        
        # optim = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        
        
        initial_learning_rate = 0.1
        final_learning_rate = 0.00001
        learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1 / 200)
        steps_per_epoch = 88

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=initial_learning_rate,
                        decay_steps=steps_per_epoch,
                        decay_rate=learning_rate_decay_factor,
                        staircase=True)
        
        # optim = tf.keras.optimizers.Adam(learning_rate=0.001)
        optim = tf.keras.optimizers.Adam()
        
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=optim,
            metrics=
            [
                'accuracy',
            ]
        )
        
        return model
        
        model = Sequential()
        
        model.add(tf.keras.layers.Conv2D(num_filters, conv_kernel_size)) #
        # push through RELU activation
        model.add(Activation('relu')) #
        # take results and run through max_pool
        model.add(MaxPooling2D(pool_size=max_pool_size)) #

        # 2nd Convolution layer
        model.add(tf.keras.layers.Conv2D(num_filters, conv_kernel_size)) # 
        model.add(Activation('relu')) #
        model.add(MaxPooling2D(pool_size=max_pool_size))

        # Fully Connected Layer
        model.add(Flatten(input_shape=(im_size, im_size)))
        model.add(Dense(128))  # Fully connected layer in Keras
        model.add(Activation('relu'))

        # dropout some neurons to reduce overfitting
        model.add(Dropout(drop_prob))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer='adam',
            metrics=
            [
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.Accuracy(),
                tf.keras.metrics.Accuracy(),
            ]
        )
        # model.summary()
        return model


def copy_script(im_size: int):
    """
    :param: im_size: image_size to build the model from
    creates and builds the model be ready for training
    :return:
    """
    # TODO - constants which need to be replaced
    train_im_count = 39775
    test_image_count = 9940
    root_dir = r'E:\aiml\burnham\sbb\dragos_initial_archive\SBB data and model\Data'
    train_set_path = os.path.join(root_dir, 'TrainSet')
    train_set_labels_path = os.path.join(root_dir, 'TrainSetLabels')
    test_set_path = os.path.join(root_dir, 'TestSet')
    test_set_labels_path = os.path.join(root_dir, 'TestSetLabels')

    # TODO - image reading which needs to be refined
    # reading the training data
    f = open(train_set_path, 'rb')
    a = (np.frombuffer(f.read(4), dtype=np.int32, count=1)).item()
    train_image_count = (np.frombuffer(f.read(4), dtype=np.int32, count=1)).item()
    image_cols = (np.frombuffer(f.read(4), dtype=np.int32, count=1)).item()
    image_rows = (np.frombuffer(f.read(4), dtype=np.int32, count=1)).item()

    x_train = np.frombuffer(
        f.read(image_rows * image_cols * train_image_count),
        dtype=np.uint8,
        count=image_rows * image_cols * train_image_count
    )
    x_train = x_train.reshape(train_image_count, image_rows, image_cols)
    x_train = x_train / 255.0
    f.close()

    f = open(train_set_labels_path, "rb")
    f.read(8)
    y_train = np.frombuffer(f.read(train_image_count), dtype=np.uint8, count=train_image_count)
    y_train = y_train.reshape(train_image_count, )
    f.close()

    # reading the validation data
    f = open(test_set_path, "rb")
    a = (np.frombuffer(f.read(4), dtype=np.int32, count=1)).item()
    test_image_count = (np.frombuffer(f.read(4), dtype=np.int32, count=1)).item()
    image_cols = (np.frombuffer(f.read(4), dtype=np.int32, count=1)).item()
    image_rows = (np.frombuffer(f.read(4), dtype=np.int32, count=1)).item()
    x_test = np.frombuffer(
        f.read(image_rows * image_cols * test_image_count),
        dtype=np.uint8,
        count=image_rows * image_cols * test_image_count
    )

    x_test = x_test.reshape(test_image_count, image_rows, image_cols)
    x_test_images = x_test  # images to be saved
    x_test = x_test / 255.0
    f.close()

    f = open(test_set_labels_path, "rb")
    f.read(8)
    y_test = np.frombuffer(f.read(test_image_count), dtype=np.uint8, count=test_image_count)
    y_test = y_test.reshape(test_image_count, )
    f.close()

    print(x_train.shape)

    # finished reading reading the training data
    x_train, x_test = x_train.reshape(-1, image_cols, image_rows, 1), x_test.reshape(-1, image_cols, image_rows, 1)
    print(x_train.shape)
    y_train, y_test = y_train.reshape(-1, ), y_test.reshape(-1, )
    print(y_train.shape)
    # Layer values
    num_filters = 32  # Number of conv filters
    max_pool_size = (2, 2)  # shape of MaxPool
    conv_kernel_size = (3, 3)  # conv kernel shape
    imag_shape = (image_cols, image_rows, 1)
    num_classes = 2
    drop_prob = 0.5  # fraction to drop (0-1.0)

    # Define the model type
    model = Sequential()

    # Define layers in the NN Define the 1st convlution layer.  We use border_mode= and input_shape only on first
    # layer border_mode=value restricts convolution to only where the input and the filter fully overlap (ie. not
    # partial overlap) model.add(tf.keras.Input(shape=imag_shape))
    model.add(tf.keras.layers.Conv2D(num_filters, conv_kernel_size))
    # push through RELU activation
    model.add(Activation('relu'))
    # take results and run through max_pool
    model.add(MaxPooling2D(pool_size=max_pool_size))

    # 2nd Convolution layer
    model.add(tf.keras.layers.Conv2D(num_filters, conv_kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=max_pool_size))

    # Fully Connected Layer
    model.add(Flatten(input_shape=(image_cols, image_rows)))
    model.add(Dense(128))  # Fully connected layer in Keras
    model.add(Activation('relu'))

    # dropout some neurons to reduce overfitting
    model.add(Dropout(drop_prob))

    # Readout layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Set loss and measurement, optimizer, and metric used to evaluate loss
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',  # was adadelta
        metrics=['accuracy'],
    )

    #   Training settings
    batch_size = 128
    num_epoch = 128

    # fit the training data to the model.  Nicely displays the time, loss, and validation accuracy on the test data

    model.fit(x_train, y_train, epochs=num_epoch)

    # save the model to disk

    # model.save(os.path.abspath(os.getcwd()) + "\savedmodel", overwrite=True)

    score, acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', acc)
