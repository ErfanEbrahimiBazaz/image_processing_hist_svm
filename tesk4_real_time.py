import cv2 as cv
import dlib
import numpy as np


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
        # print(m)  # image is tilted -0.05

        # straiting the image:
        if m != 0:
            angle = np.rad2deg(np.arctan2(y45 - y36, x45 - x36))
            return angle
        return 0


def rotate_frame(frame, angle):
    """
    Rotates an image by input angle
    :param frame: input frame to rotate
    :param angle: angle in degree to tilt frame
    :return: tilted
    """
    # straiting the frame:
    if angle != 0:
        # rotate the image to become straight
        w = frame.shape[0]
        h = frame.shape[1]
        cent = (w // 2, h // 2)
        # cent = (x36, y36)

        m = cv.getRotationMatrix2D(cent, angle, 1.0)
        rotated_img = cv.warpAffine(frame, m, cent)
        cv.imshow('rotated frame', rotated_img)
        return rotated_img

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# read the image
cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv.cvtColor(src=frame, code=cv.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # Loop through all the points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Draw a circle
            cv.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # show the image
    cv.imshow(winname="Face", mat=frame)
    angle = calculate_slope_between_eyes_input_img(frame)
    print(angle)
    if angle != 0.0:
        rotate_frame(frame, angle)

    # Exit when escape is pressed
    if cv.waitKey(delay=1) == 27:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv.destroyAllWindows()