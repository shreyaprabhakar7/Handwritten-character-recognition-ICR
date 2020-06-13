#For the preprocessing part, mainly two things were done:
#1. flipping of images - this was done by taking the transpose of the image matrix.
#2. normalization of images -  In this case, as the pixel value lies in range[0,255], it is sufficient to scale the pixel values in the range [0,1] by simply dividing the array by 255.



num_classes = 47 
img_size = 28

def img_label_load(data_path, num_classes=None):
    data = pd.read_csv(data_path, header=None)
    data_rows = len(data)
    print(data_rows)
    if not num_classes:
        num_classes = len(data[0].unique())
        print(num_classes)
 
    img_size = int(np.sqrt(len(data.iloc[0][1:])))
    print(img_size)

    imgs = np.transpose(data.values[:,1:].reshape(data_rows, img_size, img_size, 1), axes=[0,2,1,3])
    
    labels = keras.utils.to_categorical(data.values[:,0], num_classes) # one-hot encoding vectors
    
#    print(labels.shape)
#     print(imgs/255.)
    
    return imgs/255., labels
