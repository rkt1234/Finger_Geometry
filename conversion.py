#Edge detection using Canny method
import cv2
import numpy as np
import PIL
img=cv2.imread("input.jpeg") #read image
t_lower=50 #lower threshold
t_upper=150 #upper threshold

#Applying the Canny edge filter

edge=cv2.Canny(img,t_lower,t_upper)
for i in range (0,5):
    edge[i]=0
for i in range (0,200):
    for k in range (0,6):
        edge[i][-k]=0
#cv2.imshow('original',img)
#cv2.imshow('edge',edge)
cv2.imwrite("edge.jpg", edge)
#print(img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# find the contours in the edged image
contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
image_copy = img.copy()
# draw the contours on a copy of the original image
cv2.drawContours(image_copy, contours, -1, (0, 0, 255), 2)
#print(len(contours), "objects were found in this image.")
duplicate_cnt=contours
cv2.imshow("Edged image", edge)
cv2.imshow("contours", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours = max(contours, key=lambda x: cv2.contourArea(x))
#print(contours)
hull = cv2.convexHull(contours)
cv2.drawContours(edge, [hull], -1, (255, 0, 0), 2)
cv2.imwrite("connected.jpg",edge)
cv2.imshow("hull", edge)
#print(hull.shape)
points_want= hull.reshape(hull.shape[0],2)
#print(points_want)
#Trying to draw finger tips on the image
for i in points_want:
    cv2.circle(image_copy, center=(i[0], i[1]), radius=2, color=(255, 0, 0), thickness=-1)
cv2.imwrite("fingertips.jpg",image_copy)
cv2.imshow("wanted", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
#convexity defects
hull = cv2.convexHull(contours, returnPoints=False)
defects = cv2.convexityDefects(contours, hull)
#print("The defects are ")
#hprint(defects)
if defects is not None:
    cnt = 0
    size=[]
    for i in range(defects.shape[0]):  # calculate the angle
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem by triangle properties
        if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
            cnt += 1
            size.append(b)
            size.append(c)
            cv2.circle(image_copy, far, 3, [0, 255, 0], -1)
        if cnt > 0:
            cnt = cnt+1
        #cv2.putText(image_copy, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        #(start index,end index,farthestpoint,approximate distance to farthest point)

print("The length of the fingers are ")
print(size)
cv2.imwrite("defects.jpg",image_copy)
cv2.imshow("img", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()