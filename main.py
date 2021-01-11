import cv2
import matplotlib.pyplot as plt
import numpy as np

def f():
    pass

#Function to find the correct threshold
def trackingbar():
    cv2.namedWindow("Trackbar")
    cv2.resizeWindow("Trackbar", 360, 100)
    cv2.createTrackbar("Thresh1", "Trackbar", 200, 255, f )
    cv2.createTrackbar("Thresh2", "Trackbar", 200, 255, f )
    #Getting position of trackbar
    thresh1= cv2.getTrackbarPos("Thresh1", "Trackbar")
    thresh2= cv2.getTrackbarPos("Thresh2", "Trackbar")
    src= thresh1, thresh2
    return src

img= cv2.imread("text1.jpg")
width= img.shape[0]
height=img.shape[1]

#Pre processing of the image
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
TH = trackingbar()
ksize= np.ones((5,5))
blur= cv2.GaussianBlur(gray, (5,5), 1 )
_, th1= cv2.threshold(img, TH[0] ,TH[1], cv2.THRESH_BINARY)
edges= cv2.Canny(th1, TH[0], TH[1], 1)
dial= cv2.dilate(edges, ksize, iterations=2)
edges= cv2.erode(dial, ksize, iterations= 2)

#finding contour
contours= img.copy()
biggest_contour= img.copy()
con, h = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours, con, -1, (255,0,255), 3)

#Finding biggest contour
biggest= np.array([])

for i in con:
    area= cv2.contourArea(i)
    if area > 139000:
        #print(area)
        perimeter = cv2.arcLength(i,True)
        epsilon = 0.01*cv2.arcLength(i,True)
        biggest= cv2.approxPolyDP(i, epsilon, True)
        #print(biggest)
        x,y,w,h = cv2.boundingRect(biggest)
        cv2.rectangle(biggest_contour,(x,y),(x+w,y+h),(0,255,0),2)
        

#Warp Perspective for proper alignment of the image
pts1= np.float32(biggest)
pts2= np.float32([[0,0], [width, 0], [0, height], [width, height]])
M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5)
print(M)
matrix= cv2.getPerspectiveTransform(pts1,pts2)
print(matrix)
#M and matrix have same values-- use any in the next command! 
WarpColored= cv2.warpPerspective(img, M, (width, height), cv2.INTER_NEAREST) 

#Applying Adaptive Threshold on warped image
warpGray= cv2.cvtColor(WarpColored, cv2.COLOR_BGR2GRAY)
AdaptiveTh= cv2.adaptiveThreshold(warpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2)
AdaptiveTh= cv2.bitwise_not(AdaptiveTh)
AdaptiveTh= cv2.medianBlur(AdaptiveTh, 5)

#For displaying all images together
image_label= ["original", "gray", "Threshold", "Edges", "Contour", 
              "Biggest contour", "Warp", "WarpThreshold"]
image_array= [img, gray, th1, edges, contours, biggest_contour,
              WarpColored, AdaptiveTh]

for i in range (8):
    plt.subplot(2,4, i+1)
    plt.imshow(image_array[i], cmap= "Greys_r")
    plt.yticks([]), plt.xticks([])  #Removing all ticks
    plt.title(image_label[i]) 

plt.show()