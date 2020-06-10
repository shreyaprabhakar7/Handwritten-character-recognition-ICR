import cv2
import matplotlib.pyplot as plt
%matplotlib inline

image = cv2.imread('Downloads/images_checkbox_2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
plt.figure(figsize=(40,40))

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

plt.imshow(opening, cmap="gray")
