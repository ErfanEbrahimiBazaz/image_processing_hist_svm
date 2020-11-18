import cv2 as cv
import dlib
import numpy as np


# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calculate_slope_between_eyes(img_path):
    """
    Calculates the slope (in degree) between two eyes: face landmarks 36 and 45
    :param img_path: path to sample image
    :return: slope in degree
    """
    img = cv.imread(img_path, 1)
    # Convert image into grayscale
    gray = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    faces = detector(gray)
    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # calculate the slope of a line
        # face landmarks 36 and 45 are important
        x45 = landmarks.part(45).x
        y45 = landmarks.part(45).y

        x36 = landmarks.part(36).x
        y36 = landmarks.part(36).y

        # slope
        m = (y45 - y36) / (x45 - x36)
        # print(m)  # image is tilted -0.05

        # straiting the image:
        if m != 0:
            angle = np.rad2deg(np.arctan2(y45 - y36, x45 - x36))
            return angle
        return 0


def calculate_slope_between_eyes_input_img(img):
    """
    Calculates the slope (in degree) between two eyes: face landmarks 36 and 45
    :param img: input image
    :return: slope in degree
    """
    # Convert image into grayscale
    gray = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    faces = detector(gray)
    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # calculate the slope of a line
        # face landmarks 36 and 45 are important
        x45 = landmarks.part(45).x
        y45 = landmarks.part(45).y

        x36 = landmarks.part(36).x
        y36 = landmarks.part(36).y

        # slope
        m = (y45 - y36) / (x45 - x36)
        print(m)  # image is tilted -0.05

        # straiting the image:
        if m != 0:
            angle = np.rad2deg(np.arctan2(y45 - y36, x45 - x36))
            return angle
        return 0


def rotate_image(img_path, angle):
    """
    Rotates an image by input angle
    :param img: input image to rotate
    :param angle: angle in degree to tilt image
    :return: tilted
    """
    img = cv.imread(img_path, 1)
    # straiting the image:
    if angle != 0:
        # rotate the image to become straight
        w = img.shape[0]
        h = img.shape[1]
        cent = (w // 2, h // 2)
        # cent = (x36, y36)

        m = cv.getRotationMatrix2D(cent, angle, 1.0)
        rotated_img = cv.warpAffine(img, m, cent)
        return rotated_img


# read the image
img_path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses3\HWSes3\test_img.jpg'
angle = calculate_slope_between_eyes(img_path)
print(angle)

# rotate the image to make slope zero
rotated_image = rotate_image(img_path, angle)
rotated_image_angle = calculate_slope_between_eyes_input_img(rotated_image)
print('After roateating the image the angle is {}'.format(rotated_image_angle))
img = cv.imread(img_path, 1)

# Convert image into grayscale
gray = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)

# Use detector to find landmarks
faces = detector(gray)
for face in faces:
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point

    # Create landmark object
    landmarks = predictor(image=gray, box=face)

    # Loop through all the points
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        # Draw a circle
        cv.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # calculate the slope of a line
    # face landmarks 36 and 45 are important
    x45 = landmarks.part(45).x
    y45 = landmarks.part(45).y
    cv.circle(img=img, center=(x45, y45), radius=5, color=(0, 0, 255), thickness=-1)

    x36 = landmarks.part(36).x
    y36 = landmarks.part(36).y
    cv.circle(img=img, center=(x36, y36), radius=5, color=(0, 0, 255), thickness=-1)

    # slope
    m = (y45 - y36)/(x45 - x36)
    # print(m) # image is tilted -0.05

    # straiting the image:
    if m != 0:
        angle = np.rad2deg(np.arctan2(y45 - y36, x45 - x36))
        # print('angle is {}'.format(angle))

        # rotate the image to become straight
        w = img.shape[0]
        h = img.shape[1]
        cent = (w // 2, h // 2)
        # cent = (x36, y36)

        m = cv.getRotationMatrix2D(cent, angle, 1.0)
        rotated_img = cv.warpAffine(img, m, cent)

        cv.imshow('rotated image', rotated_img)

cv.imshow(winname='face image', mat=img)
cv.waitKey(0)

# Close all windows
cv.destroyAllWindows()

