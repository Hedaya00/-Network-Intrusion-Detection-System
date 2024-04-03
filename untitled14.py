import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load data
train = pd.read_csv("Train_data1.csv")
test = pd.read_csv("Test_data1.csv")
print("Training data has {} rows & {} columns".format(train.shape[0], train.shape[1]))
print("Testing data has {} rows & {} columns".format(test.shape[0], test.shape[1]))

# Data Preparation
scaler = StandardScaler()
cols = train.select_dtypes(include=['float64', 'int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64', 'int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64', 'int64']))
sc_traindf = pd.DataFrame(sc_train, columns=cols)
sc_testdf = pd.DataFrame(sc_test, columns=cols)

encoder = LabelEncoder()
cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()
traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)
enctrain = traincat.drop(['class'], axis=1)
cat_Ytrain = traincat[['class']].copy()

train_x = pd.concat([sc_traindf, enctrain], axis=1)
train_y = train['class']
test_df = pd.concat([sc_testdf, testcat], axis=1)

# Feature Selection
rfc = RandomForestClassifier()
rfe = RFE(rfc, n_features_to_select=15)
rfe = rfe.fit(train_x, train_y)
selected_features = train_x.columns[rfe.support_]

# Dataset Partition
X_train, X_test, Y_train, Y_test = train_test_split(train_x[selected_features], train_y, train_size=0.70, random_state=2)

# Model Fitting
models = {
    'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1),
    'LogisticRegression': LogisticRegression(n_jobs=-1, random_state=0),
    'BernoulliNB': BernoulliNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(criterion='entropy', random_state=0)
}

# Model Evaluation
for name, model in models.items():
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, predictions)
    confusion_matrix = metrics.confusion_matrix(Y_test, predictions)
    classification_report = metrics.classification_report(Y_test, predictions)

    print("===================================")
    print(f"{name} Performance:")
    print("===================================")
    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    print("\nClassification Report:")
    print(classification_report)
    print()
