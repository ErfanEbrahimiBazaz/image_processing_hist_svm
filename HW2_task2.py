# classifying cars by colors
import cv2 as cv
from sklearn import preprocessing
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import requests


'''
The algorithm to classify cars based on colors is as follows:
1- Use elbow meathod to calculate the number of centroids for a sample image of each category.
2- Take the maximum number for elbow
3- Define k as (maximum_number_of_elbows + 1) to make sure number of centroids are enough.
4- Scan each folder, take the name of folder (car's color) as label.
5- Apply Kmeans to all images in each category by taking k centroids.
6- Apply histogram to each processed image from last step and make feature vector for each image.
7- Make a data structure with label and values of feature vector.
8- Make a KNN classifier and train it by 90% of data, test it with 10%.
9- For KNN set number of centroids the same as K, or apply elbow method seperately to the new data structure.  
'''

# calculate KMeans of input image
def kmeans_number_of_centroids(img, path):
    '''
     calculates the minimum value of inertia (sum of squared distance), returns the index of min(inertia) which is number
     of centroids.
     :param img: name of image to process
     :param path: folder address of image
     :return: (number of centroids, all inertia values)
     '''
    img_path = os.path.join(path, img)
    img = cv.imread(img_path, 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # cv.imshow('{}'.format(img[:-4]), img)
    img = img.reshape((img.shape[0]*img.shape[1], 3))
    inertia_vals = kmeans_inertia_vals(img)
    min_inertia = np.amin(inertia_vals)
    idx_min_inertia = np.where(inertia_vals == min_inertia)
    return (idx_min_inertia[0], inertia_vals)

def kmeans_inertia_vals(img):
    '''
    deterrmine the number of centroids for an input image by using elbow method
    number of centroid is calculated by minimizing inertia. Inertia is sum of squared distance.
    By calculating 20 number of inertia, it determined which number of centroid is best for minimum inertia.
    :return: inertia values for each number of centroid
    '''
    inertia_vals = []
    for i in range(1, 21):
        print('Calculating inertia for {} number of centroids - kmeans elbow method'.format(str(i)))
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(img)
        o = kmeans.inertia_
        inertia_vals.append(o)
    return inertia_vals

def stalling_inertia(inertia_vals, accuracy):
    '''
    define when inertia (error or entropy) is stalling and therefore returns the best value
    Best inertia value is not the minimum, but the value after which the loss function does not significantly gets better.
    :param inertia_vals: list of inertia
    :param accuracy: stalling value, if loss difference between two iterations doesn't get better, stop and return index
    :return: best inertia's index
    '''
    loss = []
    for i in range(len(inertia_vals)):
        if i == len(inertia_vals) - 1:
            break
        l = inertia_vals[i] - inertia_vals[i+1]
        loss.append(l)

    loss = np.array(loss)
    for val in loss:
        if val < accuracy:
            res = np.where(loss == val)
            return res[0]
    return loss


def kmeans_preprocess(folder_path, img_name, num_centroids, dest_path):
    '''
    :param img_path: folder path where image samples are saved (cars with a specific color)
    :param num_centroids: best number of centroid calculated by elbow method
    :return: void method - saves the process image by kmeans algorith in a destination folder
    '''
    img_path = os.path.join(folder_path, img_name)
    img = cv.imread(img_path, 1)
    z = img.reshape((-1, 3))  # an array of w*h BGR values same as img.reshape((img.shape[0]*img.shape[1], 3))
    # convert to np.float32, necessary input type for cv2.kmeans()
    z = np.float32(z)

    # define criteria, iteration termination conditions
    '''
        cv.TERM_CRITERIA_EPS:  specified accuracy,
        cv.TERM_CRITERIA_MAX_ITER: specified number of iterations, max_iter, 

        cv.KMEANS_RANDOM_CENTERS: one of the two flags to chose centers for kmeans.
        cv2.KMEANS_PP_CENTERS and cv2.KMEANS_RANDOM_CENTERS
    '''
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = num_centroids

    # appling kmeans to image
    # ret: compactness
    # labels: marking each element
    # center: centroids of kmeans
    # flags: cv.KMEANS_RANDOM_CENTERS or cv.KMEANS_PP_CENTERS
    # return value (compactness, label, center)
    # compactness: sum of squared distance from each point to their corresponding distance
    # label: which cluster the point belongs to
    # centers: center of clusters which make compactness minimum
    compactness, cluster_label, center = cv.kmeans(z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[cluster_label.flatten()]
    # reshaping res from(w*h,3) to image dimension (w,h,3)
    res2 = res.reshape((img.shape))
    file_name = 'pre_' + img_name
    file_name = os.path.join(dest_path, file_name)
    cv.imwrite(file_name, res2)

def extract_features(img_path):
    '''
    calculates the histogram of an image
    :param img_path: path to input image
    :return: hostogram of the image
    '''
    img = cv.imread(img_path, 0)
    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist(images=[img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    return list(hist.squeeze())


def extract_features_hist(img_path, bins):
    """
    Calculates histogram of input color image
    :param img_path: full absolute path to image
    :param bins: number of bins to calculate histogram
    :return: histogoram as list
    """
    imag = cv.imread(img_path, 1)
    hist, bins = np.histogram(imag.flatten(), bins=bins, range=[0, 255])
    return list(hist.squeeze())


def scan_folder(path):
    '''
    extracts files from a specific folder
    :param path: input path of a folder
    :return: all images in the path as a list
    '''
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]




# sample_img_path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses3\HW_Ses3\cars\black'
# inertia = kmeans_number_of_centroids('fac51313f58b85.jpg', sample_img_path)
# # As index starts from 0, add 1 to get to the number of centroids.
# best_num_centroid = stalling_inertia(inertia[1], 1e8)
# best_num_centroids = best_num_centroid[0] + 1
# print(inertia[1])
# # plotting inertia values
# plt.plot(list(np.arange(1, 21)), inertia[1])
# plt.show()
# print('best number of centroids is {}'.format(str(best_num_centroids)))


# read all directories and set labels as name of directories -> labels of data structure
src_path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses3\HW_Ses3\cars'
dir_list = os.listdir(src_path)
dest_path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses3\HW_Ses3\kmean_cars'
labels = []

# preprocessing: applying kmeans to all cars
for fol in dir_list:
    labels.append(fol)
for label in labels:
    car_category_path = os.path.join(src_path, label)
    # print(car_category_path)
    car_images = [f for f in os.listdir(car_category_path) if os.path.isfile(os.path.join(car_category_path, f))]
    new_dest_path = os.path.join(dest_path, label)
    print(new_dest_path)
    try:
        os.mkdir(new_dest_path)
    except OSError:
        print("Creation of the directory %s failed" % new_dest_path)
    else:
        print("Successfully created the directory %s " % new_dest_path)
    for car in car_images:
        kmeans_preprocess(car_category_path, car, 5, new_dest_path)

scores = []
for neighbor in range(6, 15):
    for biin in range(5, 20):
        print('Running model for  neighbors = {}, bin = {}'.format(neighbor, biin))
        # calculate histogram (extract features for each car)
        # take the number of features the same as best number of centroids + 2
        path_kmean_cars = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses3\HW_Ses3\kmean_cars'
        dir_list = os.listdir(path_kmean_cars)
        labels = []
        feature_vector = []
        for fol in dir_list:
            labels.append(fol)
            n_path = os.path.join(path_kmean_cars, fol)
            img_list = scan_folder(n_path)
            for img in img_list:
                car_feature = extract_features_hist(os.path.join(n_path, img), biin)
                # car_feature = extract_features(img)
                # car_feature = extract_features_hist(img)
                car_feature.insert(0, fol)
                feature_vector.append(car_feature)
        # print(feature_vector[1])
        # print(len(feature_vector)) ## 491 images
        # print(feature_vector)

        # define target value (y) and data (x)
        y = [row[0] for row in feature_vector]
        X = [row[1:] for row in feature_vector]
        # print(X)
        # print(len(X))

        #split dataset into train and test data
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=1, stratify=y)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
        # make the model
        # print('Making KNN model for  neighbors = {}, bin = {}'.format(neighbor, biin))
        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=neighbor)
        # Fit the classifier to the data
        knn.fit(X_train, y_train)

        score_vals = dict()
        # test the model
        # show first 5 model predictions on the test data
        # print('First 5 tests data')
        # print(knn.predict(X_test)[0:5])
        # check accuracy of our model on the test data
        score_vals['neighbors'] = neighbor
        score_vals['bins'] = biin
        score_vals['score'] = knn.score(X_test, y_test)
        scores.append(score_vals)
        print(score_vals)

# calculate best run parameters
# print(scores)
scores_list = [record['score'] for record in scores]
# best_run = [record for record in scores if record['score'] == np.max(scores)]
best_run = [record for record in scores if record['score'] == np.max(scores_list)]
print(best_run)
# [{'neighbors': 9, 'bins': 17, 'score': 0.62}]


