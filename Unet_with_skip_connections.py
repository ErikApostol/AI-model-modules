'''
Major difference between my model and the original U-Net:

* Add a skip connection between every level of the contracting path and the corresponding level of the expanding path.
* Add dropouts after maxpooings and concatenate
* Add a scalar input in addition to image inputs.
'''

lambd = 0.00001
img_size_target = 128

def build_model(image_input, scalar_input, start_neurons):
    
    img_w_h = img_size_target

    ''' https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose
        Here is the correct formula for computing the size of the output with tf.layers.conv2d_transpose():
            # Padding==Same:
            H = H1 * stride
            # Padding==Valid
            H = (H1-1) * stride + HF
        where, H = output size, H1 = input size, HF = height of filter'''
    
    depth1= Conv2DTranspose(1, (3, 3), strides=(img_w_h, img_w_h), padding="same", kernel_regularizer=regularizers.l2(lambd))(depth_input)
    conv1 = concatenate([image_input, depth1])
    # 128 -> 64
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    depth2= Conv2DTranspose(1, (3, 3), strides=(int(img_w_h/2), int(img_w_h/2)), padding="same", kernel_regularizer=regularizers.l2(lambd))(depth_input)
    conv2 = concatenate([pool1, depth2])
    # 64 -> 32
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    depth3= Conv2DTranspose(1, (3, 3), strides=(int(img_w_h/4), int(img_w_h/4)), padding="same", kernel_regularizer=regularizers.l2(lambd))(depth_input)
    conv3 = concatenate([pool2, depth3])
    # 32 -> 16
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    depth4= Conv2DTranspose(1, (3, 3), strides=(int(img_w_h/8), int(img_w_h/8)), padding="same", kernel_regularizer=regularizers.l2(lambd))(depth_input)
    conv4 = concatenate([pool3, depth4])
    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv4)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    depth5= Conv2DTranspose(1, (3, 3), strides=(int(img_w_h/16), int(img_w_h/16)), padding="same", kernel_regularizer=regularizers.l2(lambd))(depth_input)
    conv5 = concatenate([pool4, depth5])
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv5)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(lambd))(convm)
    # add a conv layer in the skip connection conv4->uconv4
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv4)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv4)
    # add a conv layer in the skip connection conv3->uconv3
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv3)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv3)
    # add a conv layer in the skip connection conv2->uconv2
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv2)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv2)
    # add a conv layer in the skip connection conv1->uconv1
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(conv1)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lambd))(uconv1)

    #uconv1 = Dropout(0.5)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid", kernel_regularizer=regularizers.l2(lambd))(uconv1)
    
    return output_layer

image_input = Input((img_size_target, img_size_target, 2)) 
# The argument is a tuple; if the tuple has only one element, a comma should be added at the end.
scalar_input = Input((1, 1, 1))
output_layer = build_model(image_input, scalar_input, 20)  # The original version of U-Net starts with 64 channels.

model = Model(inputs=[image_input, scalar_input], outputs=output_layer)

# default: Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# adam = Adam(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

