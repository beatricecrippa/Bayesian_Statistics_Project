import cv2
import numpy as np
import pylab
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import time

scaleFac = 4
lineWidth = 4

# EXTRACT 3D MATRIX
# Set video to load
videoPath = "VIDEO1.avi"
cap = cv2.VideoCapture(videoPath)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))
data = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, frame = cap.read()
    buf[fc] = frame[:,:,0]
    data[fc] = frame[:,:,0]
    fc += 1

cap.release()
###################
#THRESHOLDING + SLICE DRAWING
# single frame selection
img = buf[50,:,:] 

# slice drawing
pts = np.array(((53,46),(34,57),(12,16),(107,17),(73,36)))
for i in range(np.size(pts,0)-1):
  cv2.line(img, tuple(pts[i,:]), tuple(pts[i+1,:]), (0, 0, 0), 3)
  
# thresholding on slice
ret, threshed_img = cv2.threshold(img,1, 255, cv2.THRESH_BINARY)
threshed_imgR = cv2.resize(threshed_img,None,fx=scaleFac,fy=scaleFac)
cv2.imshow("threshold", threshed_imgR)

# frame resize and plot
img = cv2.resize(img,None,fx=scaleFac,fy=scaleFac)
cv2.imshow("Frame", img)

# PERSPECTIVE TRANSFORMATION
#pts1 = np.float32([p1, p2, p3, p4])
#pts2 = np.float32([[0, 0], [121, 0], [0, 71], [121, 71]])
#matrix = cv2.getPerspectiveTransform(pts1, pts2)
# 
#result = cv2.warpPerspective(img, matrix, (121, 71))
#result = cv2.resize(result,None,fx=4,fy=4)
#cv2.imshow("Perspective transformation", result)

###################
# VIDEO VISUALIZER 
# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break
  # slice drawing
  for i in range(np.size(pts,0)-1):
    cv2.line(frame, tuple(pts[i,:]), tuple(pts[i+1,:]), (0, 0, 0), 3)
  
  # resize frame
  frame = cv2.resize(frame,None,fx=scaleFac,fy=scaleFac)
  # show frame
  cv2.imshow('MultiTracker', frame)
  time.sleep(.5)
   
 
  # quit on ESC button
  if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
    break
  
####################
# VISUALIZER W/ SCROLLBAR
plt.figure(2)
ax = pylab.subplot(111)
pylab.subplots_adjust(left=0.25, bottom=0.25)

frame = 0
l = pylab.imshow(data[frame,:,:],cmap='gray') 
for i in range(np.size(pts,0)):
  pylab.scatter(pts[i,0], pts[i,1], s=1, c='red', marker='o')

axcolor = 'lightgoldenrodyellow'
axframe = pylab.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
sframe = Slider(axframe, 'Frame', 0, frameCount, valinit=0)

def update(val):
    frame = int(sframe.val)
    l.set_data(data[frame,:,:])

sframe.on_changed(update)

pylab.show()

# convert the grayscale image to binary image
gray_image = data[50,:,:] 
# resize frame
gray_image = cv2.resize(gray_image,None,fx=scaleFac,fy=scaleFac)
ret,thresh = cv2.threshold(gray_image,127,255,0)
 
# find contours in the binary image
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
  # calculate moments for each contour
  M = cv2.moments(c) 
  
  # calculate x,y coordinate of center
  if M["m00"] != 0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
  else:
    cX, cY = 0, 0
     
  cv2.circle(gray_image, (cX, cY, 5, (255, 255, 255), -1))  
  cv2.putText(gray_image, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  
  # display the image
  cv2.imshow("Image", gray_image)
  cv2.waitKey(0)
  # quit on ESC button
  if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
    break


#FRAME SPEGNIMENTO LASER: 185
