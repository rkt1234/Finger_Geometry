 #Edge detection using Canny method
import cv2
import numpy as np
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
# cv2.imshow('original',img)
# cv2.imshow('edge',edge)
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
cnt=contours
cv2.imshow("Edged image", edge)
cv2.imshow("contours", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours = max(contours, key=lambda x: cv2.contourArea(x))
print(contours)
hull = cv2.convexHull(contours)
cv2.drawContours(edge, [hull], -1, (255, 0, 0), 2)
cv2.imwrite("connected.jpg",edge)
cv2.imshow("hull", edge)
#print(hull.shape)
points_want= hull.reshape(hull.shape[0],2)
#print(points_want)
#Trying to draw finger tips on the image
#cv2.circle(image_copy, center=(points_want[0][0], points_want[0][1]), radius=2, color=(255, 0, 0), thickness=-1)
top_point=[]
for i in range(0,len(points_want)):
    
    if np.abs(points_want[i][0] - points_want[i-1][0]) >= 10 or np.abs(points_want[i][1] - points_want[i-1][1])>=10:
        top_point.append(list(points_want[i]))
        cv2.circle(image_copy, center=(points_want[i][0], points_want[i][1]), radius=2, color=(255, 0, 0), thickness=-1)
#cv2.circle(image_copy, center=(points_want[9][0], points_want[9][1]), radius=2, color=(255, 0, 0), thickness=-1)
#print(finalpoint)
cv2.imwrite("fingertips.jpg",image_copy)
cv2.imshow("wanted", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
#convexity defects
hull = cv2.convexHull(contours, returnPoints=False)
defects = cv2.convexityDefects(contours, hull)
#print("The defects are ")
#print(defects)
if defects is not None:
    cnt = 0
    size=[]
    bottom_points=[]
    for i in range(defects.shape[0]): 
        # calculate the angle
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
            bottom_points.append(list(far))
        if cnt > 0:
            cnt = cnt+1
        #cv2.putText(image_copy, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        #(start index,end index,farthestpoint,approximate distance to farthest point)

#print("The length of the fingers are ")
#print(size, cnt)
for i in  range(1,len(top_point)): #eleiminating unnecessary point
    if top_point[i-1][1]==top_point[i][1] and np.abs(top_point[i-1][0]-top_point[i][0])>=5:
        top_point.remove(top_point[i])
        top_point.remove(top_point[i-1])
        break  
print(top_point)
print(len(top_point))
print(bottom_points)
print(len(bottom_points))
cv2.imwrite("defects.jpg",image_copy)
cv2.imshow("img", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in top_point:
    cv2.circle(image_copy,(i[0],i[1]),3,[0, 255,255],-1)
cv2.imshow("check",image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
# bottom_point anticlockwise 
# top_point clockwise
line=cv2.line(image_copy,bottom_points[0],top_point[4],[0,214,23],1)
cv2.imwrite("line.jpg",line)
cv2.imshow("line",image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
length=[]
#proper length calculation
for i in range(4):#bottom
    mini=2000
    for k in range(5):#top
        d=np.sqrt(((bottom_points[i][1]-top_point[k][1])**2)+((bottom_points[i][0]-top_point[k][0])**2))
        if d<mini:
            mini=d
    length.append(mini)
d=np.sqrt(((bottom_points[0][1]-top_point[4][1])**2)+((bottom_points[0][0]-top_point[4][0])**2))
length.append(d)
print("The finger lengths are")
print(length)
T =top_point#[[156, 51], [217, 159], [12, 103], [61, 55], [97, 35]]
T.sort(reverse=True)
#print(T)
B = bottom_points#[[93, 127], [67, 140], [139, 169], [113, 127]]
B.sort(reverse=True)
#print(B)
L =[]
for i in range(4):
    l = np.sqrt((T[i][0]-B[i][0])**2 + (T[i][1]-B[i][1])**2) 
    L.append(l)
l= np.sqrt((T[4][0]-B[3][0])**2 + (T[4][1]-B[3][1])**2)
L.append(l)

print("Length of Fingers:")
print("Thumb-->", L[0])
print("Index-->", L[1])
print("Middle-->", L[2])
print("Ring-->", L[3])
print("Little-->", L[4])   
print("2D:4D Digit Ratio:", (L[1]/L[3]))
## Finding the ratio of fingers with longest length
longest= max(L)
print(longest)
print("Ratio between fingers and the longest finger length:")
print("Thumb-->", L[0]/longest)
print("Index-->", L[1]/longest)
print("Middle-->", L[2]/longest)
print("Ring-->", L[3]/longest)
print("Little-->", L[4]/longest)   
