def CNN(input_shape,num_kernels=[20,20],kernel_shapes=[(3,3),(3,3)],dense_nodes=[128],dropout_chance=0.4,num_classes=47,produce_output=True):
    # A 0 inserted either in num_kernels or in dense_nodes means a Dropout layer is to be inserted at that point
    # If it is inserted in the convolutional layers, then some value must be adde in the corresponding place in kernel_shapes
    model = Sequential()
    model.add(Conv2D(num_kernels[0],kernel_size=kernel_shapes[0],activation='relu',input_shape=input_shape))
    num_conv_layers = len(num_kernels)
    for i in range(num_conv_layers-1):
        if num_kernels[i+1]==0:
            model.add(Dropout(dropout_chance))
        else:
            model.add(Conv2D(num_kernels[i+1],kernel_size=kernel_shapes[i+1],activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    num_dense_layers = len(dense_nodes)
    model.add(Flatten())
    for j in range(num_dense_layers):
        if dense_nodes[j]==0:
            model.add(Dropout(dropout_chance))
        else:
            model.add(Dense(dense_nodes[j],activation='relu'))
    if produce_output==True:
        model.add(Dense(num_classes,activation='softmax'))
    return model



cnn_model_do = CNN((28,28,1),num_kernels=[20,30],kernel_shapes=[(3,3),(4,4)],dense_nodes=[0,256,0,128,32])

# Compile and fit model
cnn_model_do.compile(loss = 'categorical_crossentropy',optimizer = 'rmsprop',metrics = ['accuracy'])
cnn_history_do = cnn_model_do.fit(X,to_categorical(y),nb_epoch = 10,validation_split = 0.2,batch_size = 128,verbose = 1)
