import numpy as np
import pandas as pd
import requests
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler



'''
1- Download images from Internet by usin scripts.
2- Images are for day and night.
3- Use the same number of images to train the model, and the same size.
4- Preprocess images by using either kmeans or geometric transformation (log transform of task 1)
5- Calculate histogram and determine the feature vector.
6- Train an SVM model.
7- Calculate the accuracy of trained model.
'''

def log_transform(img_path, img_name, path):
    """
    Calculates the log transform of an input image
    :param img_path: absolute path to image
    :param img_name: name of transformed image to be saved
    :param path: where to save the transformed image
    :return: log transformed image
    """
    img_path = os.path.join(img_path, img_name)
    img = cv.imread(img_path, 1)
    # defining parameters
    # best value for c
    c = 255 / np.log(1 + np.max(img))
    gamma = 0.5
    transformed_img = c * (img ** gamma)
    transformed_img = np.array(transformed_img, dtype=np.uint8)

    cv.imwrite(os.path.join(path, img_name), transformed_img)
    return transformed_img


def save_img(dest_path, name, img):
    """
    Save an input image to a file
    :param dest_path: where to save image
    :param name: name of output file
    :param name: input image to save
    :return: void method, saves the result on HDD
    """
    img_name = os.path.join(dest_path, name)
    cv.imwrite(img_name, img)


def resize_image(img, width, height, interpolation = cv.INTER_AREA ):
    """
    Resize an image to predetermined values
    :param img: input image to resize
    :return: resized image
    """
    img_resize = cv.resize(img,(width,height),interpolation=interpolation)
    return img_resize


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


def download_images(df, photo_df, path, filter_word):
    """
    Download images by filtering data collections and making HTTP GET requests to https://unsplash.com/ web page
    :param df: data['collections'] from https://unsplash.com/
    :param photo_df: datasets['photos']
    :param path: path to save images
    :param filter_word: key word to filter image data set
    :return: void method - the result is saved in path
    """
    filtered_col = df[df['collection_title'].isin([item for item in [col for col in df['collection_title']] if filter_word in str(item)])]
    df_day = pd.merge(filtered_col, photo_df, how='inner', left_on='photo_id', right_on='photo_id' )[['photo_id', 'collection_title', 'photo_image_url','ai_description']]

    for index, row in df_day.iterrows():
        img_name = row['collection_title']
        img_url = row['photo_image_url']
        #dropping special characters from image name
        img_new_name = ''.join(e for e in img_name if e.isalnum())
        img_new_name += '.png'
        receive = requests.get(img_url)
        path = path
        download_path = os.path.join(path, img_new_name)
        with open(download_path,'wb') as f:
            f.write(receive.content)


# read all directories and set labels as name of directories -> labels of data structure
src_path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses3\HW_Ses3\daynight'
dir_list = os.listdir(src_path)
dest_path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses3\HW_Ses3\task3_prcimgs'
labels = []
# preprocessing: applying log transformation to all cars
# for fol in dir_list:
#     labels.append(fol)
# for label in labels:
#     day_night_path = os.path.join(src_path, label)
#     # print(car_category_path)
#     day_night_images = [f for f in os.listdir(day_night_path) if os.path.isfile(os.path.join(day_night_path, f))]
#     new_dest_path = os.path.join(dest_path, label)
#     print(new_dest_path)
#     try:
#         os.mkdir(new_dest_path)
#     except OSError:
#         print("Creation of the directory %s failed" % new_dest_path)
#     else:
#         print("Successfully created the directory %s " % new_dest_path)
#     for img in day_night_images:
#         print(img)
#         print(new_dest_path)
#         log_transform(day_night_path, img, new_dest_path)

# extract feature vector by histogram
scores = []
for biin in range(3, 18):
    # calculate histogram (extract features for each car)
    # take the number of features for a range of different bins, choose the best number of bins by calculating
    # final SVM model score
    print('Starting feature extraction for bin = {}'.format(biin))
    path_process_img = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses3\HW_Ses3\task3_prcimgs'
    dir_list = os.listdir(path_process_img)
    labels = []
    feature_vector = []
    for fol in dir_list:
        labels.append(fol)

        n_path = os.path.join(path_process_img, fol)
        img_list = scan_folder(n_path)
        for img in img_list:
            img_feature = extract_features_hist(os.path.join(n_path, img), biin)
            # car_feature = extract_features(img)
            # car_feature = extract_features_hist(img)
            img_feature.insert(0, fol)
            feature_vector.append(img_feature)
    print(feature_vector[1])
    print(len(feature_vector)) ## 491 images
    print(feature_vector)

    # define target value (y) and data (x)
    y = [row[0] for row in feature_vector]
    X = [row[1:] for row in feature_vector]
    print(X)
    print(len(X))
    print('y is {}'.format(y))

    # scaling feature data to make SVM model training much faster
    scaler = MinMaxScaler()
    scaler.fit(X)
    MinMaxScaler()
    X = scaler.transform(X)

    # split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    print('Start training model for bin = {}'.format(biin))
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    print('Start predicting by trianed model for {} feature...'.format(biin))
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    score_vals = dict()
    # test the model
    # check accuracy of our model on the test data
    score_vals['bins'] = biin
    score_vals['score'] = metrics.accuracy_score(y_test, y_pred)
    scores.append(score_vals)
    print(score_vals)

# calculate best run parameters
# print(scores)
scores_list = [record['score'] for record in scores]
# best_run = [record for record in scores if record['score'] == np.max(scores)]
best_run = [record for record in scores if record['score'] == np.max(scores_list)]
print(best_run)





