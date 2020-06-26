#change the path best suitable

%matplotlib inline
import cv2
import pandas as pd 
import numpy as np
import xml.etree.ElementTree as ET
import pytesseract as pytess
import sys
from models import get_model_config, get_model_name, load_model
from utils import ask_for_file_particulars,get_white_foreground_and_black_background
import cv2
import skimage.io as skio
from skimage.transform import rescale, resize
from decimal import Decimal, ROUND_HALF_UP
from preprocessing import preprocess
import matplotlib.pyplot as plt

dumps = list()
annotations=template_file
in_file = open(template_file)
tree=ET.parse(in_file)
root = tree.getroot()
jpg = annotations.split('.')[0] + '.jpg'
imsize = root.find('size')
w = int(imsize.find('width').text)
h = int(imsize.find('height').text)
all = list()

for obj in root.iter('object'):
        current = list()
        name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        xn = int(float(xmlbox.find('xmin').text))
        xx = int(float(xmlbox.find('xmax').text))
        yn = int(float(xmlbox.find('ymin').text))
        yx = int(float(xmlbox.find('ymax').text))
        current += [jpg,w,h,name,xn,yn,xx,yx]
        all += [current]

in_file.close()

data = pd.DataFrame(all,columns=['path','width','height','label','xmin','ymin','xmax','ymax'])

# Read input image
input_image = cv2.imread('/content/gdrive/My Drive/icr/form-extractor-ocr/data/input_forms/form_sample01.jpg')

# #Image pre-processing
input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
preprocess(input_image)
img = cv2.imread('/content/gdrive/My Drive/icr/form-extractor-ocr/data/temp/output.png')

class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
def predict(model_config, file_location):
    model = load_model(model_config['filepath_weight'], model_config['filepath_architechture'])

    # images format conversion
    a = []
    img = skio.imread(file_location)
    img = resize(img, (16, 8))
    img = img.tolist()
    a.append(img)
    img = np.asarray(a)
    x_test = img

    # Confidence of all alphabets
    prediction = model.predict(x_test, batch_size=32, verbose=0)
    result = np.argmax(prediction, axis=1)
    result = result.tolist()
    for i in prediction:
      confidence = prediction[0][result]

    result_alphabet = [class_mapping[int(x)] for x in result]
    confidence= Decimal(confidence[0]*100)

    confidence = Decimal(confidence.quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
    return result_alphabet[0], confidence
    
    model_config = get_model_config('larger_CNN')
model_config['filepath_weight']='/content/gdrive/My Drive/icr/form-extractor-ocr/data/models/larger_CNN_weight'
model_config['filepath_architechture']='/content/gdrive/My Drive/icr/form-extractor-ocr/data/models/larger_CNN_model'
for i,row in data.iterrows():
    x1=row['xmin']
    y1=row['ymin']
    x2=row['xmax']
    y2=row['ymax']
    image=img[y1:y2,x1:x2]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh, bnw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    crop_image_path='/content/gdrive/My Drive/icr/form-extractor-ocr/data/crop_image/show_image'+str(i)+'.png'
    cv2.imwrite(crop_image_path,bnw)
    result_alphabet, confidence = predict(model_config, crop_image_path)
    if confidence>80:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.putText(img,result_alphabet,(x1+10,y1-5), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(img,result_alphabet,(x1+10,y1-5), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2,cv2.LINE_AA)

    print('The model predicts alphabet: {} with {}% confidence'.format(result_alphabet, confidence))
    
    cv2.imwrite('/content/gdrive/My Drive/icr/form-extractor-ocr/data/output/final.png',img)
print("Final output saved.")
#Display input images and output depth map
f = plt.figure(figsize=(30,50)) 
f.add_subplot(1,1, 1)
plt.title("Labelled output")
plt.imshow(img)
