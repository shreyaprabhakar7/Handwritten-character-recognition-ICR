## HANDWRITTEN TEXT RECOGNITION

In this project we are basically focusing on Handwritten character recignition in official documents containing characters in textboxes.
we will be using the form/document images present in the images section of this repo.

#### STEPS FOLLOWED ARE:

PREPROCESSING : 

1. dewarping of images - This is done to remove the skewness of the document.

2. denoising of images - There are various noises that are required to be removed before performing the actual extraction task.
                       - some of the denoising tehniques applied were:
                       1. addding gaussian blur
                       2. thresholding
                       3. opening (erosion followed by dilation)
                       
TEXTBOX EXTRACTION :

Separate characters were extracted by extracting the whole rectangular textbox containing various handwritten characters.
These extracted characters(boxes) will be then used for the testing purposes.

EMNIST TRAINING :

EMNIST stands for Extended-MNIST. It is basically a dataset containing both letters and digits. In this project, EMNIST balanced dataset is used  for training and validation purposes.
