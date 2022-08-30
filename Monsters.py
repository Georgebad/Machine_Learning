import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import timeit
import numpy as np

start = timeit.default_timer()

# Load the csv file
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv('test.csv')
ids = data_test['id']
# Formatting Data

# Replace the values of the color and type  variables with the corresponding numbers and drop the id column from both
# train and test dataframes
colors_d = {"white": 1, "black": 2, "clear": 3, "blue": 4, "green": 5, "blood": 6}
types_d = {'Ghost': 0, 'Goblin': 1, 'Ghoul': 2}
types_di = {0: 'Ghost', 1: ' Goblin', 2: 'Ghoul'}

data_train.replace({"color": colors_d}, inplace=True)
data_train.replace({"type": types_d}, inplace=True)
data_train.drop('id', axis=1, inplace=True)

data_test.replace({"color": colors_d}, inplace=True)
data_test.replace({"type": types_d}, inplace=True)
data_test.drop('id', axis=1, inplace=True)

# Split the data from train dataframe
df = pd.get_dummies(data_train.drop('type', axis=1))
X_train, X_test, y_train, y_test = train_test_split(df, data_train['type'], test_size=0.25, random_state=0)

# KNN Classification
print('\nKNN CLASSIFICATION RESULTS\n')
knn_list = [1, 3, 5, 10]
for i in knn_list:
    knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    df = pd.DataFrame(y_pred, columns=['type'])

    # Print the submission file
    y_res = knn.predict(data_test)
    df_res = pd.DataFrame(y_res, columns=['type'])
    df_res.replace({"type": types_di}, inplace=True)
    df_res.insert(0, column='id', value=ids)
    df_res.to_csv('submission'+str(i)+'.csv', index=False)
    
    # statistics
    acc_score = accuracy_score(y_test, df)
    f1 = f1_score(y_test, df, average='weighted')
    print("Accuracy score for", i, "neighbor is:", acc_score)
    print("F1 score for", i, "neighbor is:", f1)
    print("------------------------------------------------")

# Neural Networks classification
print("\nNEURAL NETWORKS CLASSIFICATION RESULTS\n")
k1_list = [50, 100, 200]

scalar = StandardScaler()
X_train1 = scalar.fit_transform(X_train)
X_test1 = scalar.fit_transform(X_test)

# MLP classification with 1 hidden level
print('Neural Networks classification with 1 hidden level:\n')
for k in k1_list:
    mlp_sigmoid = MLPClassifier(activation='logistic', hidden_layer_sizes=(k,), solver='sgd', max_iter=1500)
    mlp_sigmoid.fit(X_train1, y_train)
    y_pred_sigmoid_k = mlp_sigmoid.predict_proba(X_test1)
    res = []
    for x in y_pred_sigmoid_k:
        res.append(x.argmax())
    df_sigmoid_k = pd.DataFrame(res, columns=['type'])
    acc_score_sigmoid_k = accuracy_score(y_test, df_sigmoid_k)
    f1_sigmoid_k = f1_score(y_test, df_sigmoid_k, average='weighted')

    # Print the submission file
    y_res = mlp_sigmoid.predict(data_test)
    df_res = pd.DataFrame(y_res, columns=['type'])
    df_res.replace({"type": types_di}, inplace=True)
    df_res.insert(0, column='id', value=ids)
    df_res.to_csv('submissionMLP1_'+str(k)+'.csv', index=False)

    print("Accuracy score for K=", k, "is:", acc_score_sigmoid_k)
    print("F1 score for K=", k, " is:", f1_sigmoid_k)
    print("------------------------------------------------")

# MLP classification with 2 hidden level
k2_list = [25, 50, 100]
print('\nNeural Networks classification with 2 hidden level:\n')
for k1, k2 in zip(k1_list, k2_list):
    mlp_sigmoid_2k = MLPClassifier(activation='logistic', hidden_layer_sizes=(k1, k2,), solver='sgd')
    mlp_sigmoid_2k.fit(X_train1, y_train)
    y_pred_sigmoid_2k = mlp_sigmoid_2k.predict_proba(X_test1)
    res = []
    for x in y_pred_sigmoid_2k:
        res.append(x.argmax())
    df_sigmoid_2k = pd.DataFrame(res, columns=['type'])
    acc_score_sigmoid_2k = accuracy_score(y_test, df_sigmoid_2k)
    f1_sigmoid_2k = f1_score(y_test, df_sigmoid_2k, average='weighted')

    # Print the submission file
    y_res = mlp_sigmoid_2k.predict(data_test)
    df_res = pd.DataFrame(y_res, columns=['type'])
    df_res.replace({"type": types_di}, inplace=True)
    df_res.insert(0, column='id', value=ids)
    df_res.to_csv('submissionMLP2_'+str(k1)+'_'+str(k2)+'.csv', index=False)

    print("Accuracy score for K1=", k1, "and K2=", k2, "is:", acc_score_sigmoid_2k)
    print("F1 score for K1=", k1, "and K2=", k2, " is:", f1_sigmoid_2k)
    print("------------------------------------------------")

# SVM classification
print("\nSVM CLASSIFICATION RESULTS\n")

# SVM classification with linear kernel
svm_class_linear = svm.SVC(kernel='linear', decision_function_shape='ovr')
svm_class_linear.fit(X_train, y_train)
y_pred_linear = svm_class_linear.predict(X_test)
df_linear = pd.DataFrame(y_pred_linear, columns=['type'])
acc_score_linear = accuracy_score(y_test, df_linear)
f1_linear = f1_score(y_test, df_linear, average='weighted')

# Print the submission file
y_res = svm_class_linear.predict(data_test)
df_res = pd.DataFrame(y_res, columns=['type'])
df_res.replace({"type": types_di}, inplace=True)
df_res.insert(0, column='id', value=ids)
df_res.to_csv('submissionSVML.csv',index=False)

# Statistics
print('SVM classification with linear kernel:')
print("Accuracy score for linear kernel is:", acc_score_linear)
print("F1 score for linear kernel neighbor is:", f1_linear)
print("------------------------------------------------")

# SVM classification with RBF kernel
svm_class_rbf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
svm_class_rbf.fit(X_train, y_train)
y_pred_rbf = svm_class_rbf.predict(X_test)
df_rbf = pd.DataFrame(y_pred_rbf, columns=['type'])
acc_score_rbf = accuracy_score(y_test, df_rbf)
f1_rbf = f1_score(y_test, df_rbf, average='weighted')

# Print the submission file
y_res = svm_class_rbf.predict(data_test)
df_res = pd.DataFrame(y_res, columns=['type'])
df_res.replace({"type": types_di}, inplace=True)
df_res.insert(0, column='id', value=ids)
df_res.to_csv('submissionSVMR.csv',index=False)

# Statistics
print('SVM classification with rbf kernel:')
print("Accuracy score for rbf kernel is:", acc_score_rbf)
print("F1 score for  rbf kernel is:", f1_rbf)
print("------------------------------------------------")


# Naive-Bayes classification
class NBClassifier:
    def __init__(self):
        self.prob = None
        self.mean = None
        self.var = None
        self.classes = None
        self.count = None
        self.rows = None
        self.grouped = None
        self.results = None

    # Fit the data
    def fit(self, data, target):
        data = data.drop('color', 1)
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.rows = data.shape[0]
        self.grouped = data.groupby(target)

        self.mean_var()
        self.probability()

    # Calculate the mean and var for each characteristic
    def mean_var(self):
        self.mean = self.grouped.apply(np.mean).to_numpy()
        self.var = self.grouped.apply(np.var).to_numpy()

        return self.mean, self.var

    # Calculate the probability of each characteristic
    def probability(self):
        self.prob = (self.grouped.apply(lambda z: len(z)) / self.rows).to_numpy()
        return self.prob

    # Predict the result
    def predict(self, data_for_test):
        predictions = [self.result(f) for f in data_for_test.to_numpy()]
        return predictions

    # Calculate the probability of the result given a specific characteristic with Gaussian_density and
    # multinomial_distribution
    def result(self, y):
        self.results = []
        for j in range(self.count):
            first = np.log(self.prob[j])
            second = np.sum(np.log(self.gaussian(j, y[0:4])))
            proba = first + second
            self.results.append(proba)
        self.multinomial(y)
        return self.classes[np.argmax(self.results)]

    def gaussian(self, class_idx, z):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        up = np.exp((-1 / 2) * (((z - mean) ** 2) / (var ** 2)))
        down = np.sqrt(2 * np.pi) * var
        prob = up / down
        return prob

    def multinomial(self, y):
        first = np.log(self.prob[self.count - 1])
        y = y.astype(float)
        y /= y.sum()
        second = np.sum(np.log(np.random.multinomial(self.count, y)))
        proba = first + second
        self.results.append(proba)


print("\nNAIVE BAYS CLASSIFICATION RESULTS\n")
naive_class = NBClassifier()
naive_class.fit(X_train, y_train)
y_naive = naive_class.predict(X_test)

df_naive = pd.DataFrame(y_naive, columns=['type'])
acc_score_naive = accuracy_score(y_test, df_naive)
f1_naive = f1_score(y_test, df_naive, average='weighted')

# Print the submission file
y_res = naive_class.predict(data_test)
df_res = pd.DataFrame(y_res, columns=['type'])
df_res.replace({"type": types_di}, inplace=True)
df_res.insert(0, column='id', value=ids)
df_res.to_csv('submissionNB.csv', index=False)

# Statistics
print("Accuracy score for Naive Bayes classifier is:", acc_score_naive)
print("F1 score for Naive Bayes classifier is:", f1_naive)
print("------------------------------------------------")

stop = timeit.default_timer()
print('Time: ', stop - start)
