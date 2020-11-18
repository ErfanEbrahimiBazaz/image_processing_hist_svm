# geometrical transformation
import cv2 as cv
import numpy as np


'''
Implementing log transform
'''
img_path = 'C:\\Users\\E17538\\OneDrive - Uniper SE\\Desktop\\DailyActivities\\Data_Management\\FAD\\2011\\img.jpg'
img = cv.imread(img_path, 1)
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('original image', img)

# defining parameters
# best value for c
c = 255 / np.log(1 + np.max(img))
gamma = 0.5
transformed_img = c * (img ** gamma)

transformed_img = np.array(transformed_img, dtype=np.uint8)

cv.imshow('transformed image', transformed_img)
cv.waitKey(0)