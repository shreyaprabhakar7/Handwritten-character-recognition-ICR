import os
directory = r"C:/cropped_contours_rect13/"
ptr1=0
for filename in os.listdir(directory):
    ptr1=ptr1+1
    
print("total box extracted = " , ptr1)

import PIL
directory = r"C:/cropped_contours_rect13/"
cnt=0
lp=0
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        img=cv2.imread(os.path.join(directory, filename))
#         print(img)
        cnt=cnt+1
        ptr=0
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,bw_img = cv2.threshold(img_gray,45,255,cv2.THRESH_BINARY)
        inv_bw_img = 255-bw_img
#         print(inv_bw_img.shape)
#         print(cnt)
#         print(inv_bw_img.sum())
        if inv_bw_img.sum()>0:
            ptr=ptr+1
#         plt.imshow(inv_bw_img,cmap='gray')
#         # print(inv_bw_img)
#         cnt=0
#         for i in inv_bw_img:
#             print(i)
#             print(i.shape)
# #             print(i.sum())
#             if i.sum()>0:
#                 cnt = cnt+1
# #             else:
# #                 continue
        if ptr>0:
#             print("textbox is filled")
            lp=lp+1
#         else:
#             print("textbox is empty")


print("total textboxes in document= ",ptr1)
print("total empty textboxes (in actaul)= 600")                                          # do it manually
print("no. of filled textboxes (predicted) = ", lp)
print("no.of empty textboxes (predicted) = ", ptr1-lp)
acc = ((600 - (ptr1-lp))/600)*100
acc1 = 100-acc
print("acc =",acc1 )
            
            


# # os.chdir('C:/cropped_contours_rect8/')
# files = os.listdir('C:/cropped_contours_rect8/')

# for num, x in enumerate(files):
#         img = PIL.Image.open(x)
# #         plt.subplot(7,6,num+1)
# #         plt.title(num)
# #         plt.axis('off')
# #         plt.imshow(img)
#         print(x)
        
