from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import image
import numpy as np
import random
import timeit
import os

start = timeit.default_timer()
# set k=10 and find 10 random folders to load their images. First check if the folder contains
# at least 50 images, else move to the next
k = 10
d = 4096
random_list = []
for i in range(0, k):
    temp_num = random.randint(0, 3999)
    folder = "./train_data/" + str(temp_num)
    while len(os.listdir(folder)) <= 50:
        temp_num += 1
        folder = "./train_data/" + str(temp_num)
    random_list.append(temp_num)

# Load the images and convert them into grayscale before save them into the list
images = pd.DataFrame([])
results = []
for i in range(0, k):
    folder = "./train_data/" + str(random_list[i])
    counter = 0
    for filename in os.listdir(folder):
        if counter < 50:
            loaded = image.imread(folder + "/" + filename)
            red = loaded[:, :, 0]
            green = loaded[:, :, 1]
            blue = loaded[:, :, 2]
            gray = (0.299 * red + 0.587 * green + 0.114 * blue)
            new_face = pd.Series(gray.flatten(), name=random_list[i])
            images = images.append(new_face)
            results.append(random_list[i])
            counter += 1

results_df = pd.DataFrame(results)
X_train, X_test, y_train, y_test = train_test_split(images, results_df, test_size=0.1, random_state=20)

M = [100, 50, 25]
for i in M:
    # Reduce Data with PCA
    my_pca = PCA(n_components=i)
    my_pca.fit(images)
    res = my_pca.transform(images)
    reduced_data_df = pd.DataFrame(images)
    X_train, X_test, y_train, y_test = train_test_split(reduced_data_df, results_df, test_size=0.1, random_state=20)

    # Kmeans clustering with euclidean distance and print purity and F-measure
    kmeans_eucl = KMeans(n_clusters=10)
    kmeans_eucl.fit(X_train)
    pred = kmeans_eucl.predict(X_test)
    df = pd.DataFrame(pred, columns=['category'])
    matrix = metrics.cluster.contingency_matrix(y_test, df)
    purity = np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
    f1 = metrics.f1_score(y_test, df, average='weighted')
    print("\nPurity for Kmeans with Euclidian distance and M = ", i, " is:", purity)
    print("F-measure for Kmeans with Euclidian distance and M = ", i, " is:", f1)

    # Kmeans clustering with cosine distance and print purity and F-measure
    kmeans_cos = KMeans(n_clusters=10)
    kmeans_cos.fit(preprocessing.normalize(X_train))
    pred = kmeans_cos.predict(X_test)
    df = pd.DataFrame(pred, columns=['category'])
    matrix = metrics.cluster.contingency_matrix(y_test, df)
    purity = np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
    f1 = metrics.f1_score(y_test, df, average='weighted')
    print("Purity for Kmeans with cosine distance and M = ", i, " is:", purity)
    print("F-measure for Kmeans with cosine distance and M = ", i, " is:", f1)

    # AgglomerativeClustering clustering and print purity and F-measure
    agglo = AgglomerativeClustering(n_clusters=10, linkage="ward")
    # agglo.fit(X_train)
    pred = agglo.fit_predict(X_test)
    df = pd.DataFrame(pred, columns=['category'])
    matrix = metrics.cluster.contingency_matrix(y_test, df)
    purity = np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
    f1 = metrics.f1_score(y_test, df, average='weighted')
    print("Purity for AgglomerativeClustering with cosine distance and M = ", i, " is:", purity)
    print("F-measure for AgglomerativeClustering with cosine distance and M = ", i, " is:", f1)

# Reduce images with auto encoder
# for i in M:
#    my_encoder = MLPRegressor(alpha=d-d/4-i-d/4-d, random_state=1,max_iter=2000)
#    my_encoder.fit(X_train, X_train)


stop = timeit.default_timer()
print('Time: ', stop - start)
