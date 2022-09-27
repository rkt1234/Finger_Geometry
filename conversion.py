#Edge detection using Canny method
import cv2
img=cv2.imread("sample.jpg") #read image
t_lower=50 #lower threshold
t_upper=150 #upper threshold

#Applying the Canny edge filter
edge=cv2.Canny(img,t_lower,t_upper)
cv2.imshow('original',img)
cv2.imshow('edge',edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
