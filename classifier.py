# ---------------- HOW TO RUN THIS PYTHON SCRIPT --------------
# OBTAIN studentInfo.csv:
    # Download dataset from https://analyse.kmi.open.ac.uk/open_dataset
    # From the anonymisedData folder, move studentInfo.csv into the same directory as this file
# ENSURE THE FOLLOWING LIBRARIES ARE INSTALLED, OTHERWISE INSTALL USING FOLLOWING COMMAND
    # pandas: pip install pandas
    # seaborn: pip install seaborn
    # matplotlib: pip install matplotlib
    # numpy: pip install numpy
    # sklearn: pip install scikit-learn
# RUNNING THE CODE
    # In the terminal, in the directory containing the code, write command 'python classifer.py'
    # Graphs will be saved in the current directory

# ----------------- RANDOM FOREST CLASSIFICATION --------------
#DATA PREPARATION
import pandas as pd
# Read in data from studentInfo.csv
studentData = pd.read_csv('studentInfo.csv')
# Count number of each final_result
print(studentData.groupby("final_result").count())

# Remove features id_student and gender
studentData.drop(columns= ["id_student"], axis=1, inplace=True)
studentData.drop(columns= ["gender"], axis=1, inplace=True)

# Remove student entries where results are "Fail" or "Distinction"
# This is because we are going to be doing binary classification
# Between Pass and Withdrawn
studentData = studentData[studentData.final_result != "Distinction"]
studentData = studentData[studentData.final_result != "Fail"]

# Heatmap to show where data is missing
import seaborn as sns
import matplotlib.pyplot as plt
plt.tight_layout()
sns.heatmap(studentData.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Dropping rows with missing data. Heatmap shows there is some data missing for imd_band
studentData = studentData.dropna()

# Comment out following block of code if original imbalanced Pass and Withdrawn data is wanted, rather than filtered
# balanced Pass and Withdrawn data
count = 0
i = 0
iList = []
while count < 1910:
    if studentData.iloc[i][9] == "Pass":
        studentData.drop(index=studentData.index[i], inplace = True)
        iList.append(studentData.index[i])
        count += 1
        i+=1
    else:
        i += 1


# Replace values 'Fail' and 'Pass' in final_result with 0 and 1 respectively
studentData["final_result"] = studentData["final_result"].replace("Withdrawn", 0)
studentData["final_result"] = studentData["final_result"].replace("Pass", 1)

# One-hot encode the data using pandas get_dummies, so no categorical columns
studentDataEnc = pd.get_dummies(studentData)

# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(studentDataEnc['final_result'])

# Remove the labels from the features
# axis 1 refers to the columns
features= studentDataEnc.drop('final_result', axis = 1)
# Saving feature names
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

#DATA VISUALISATION -- UNCOMMENT THE BELOW CODE TO PLOT GRAPHS
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
#
# # Countplot of 0 (fail) vs 1 (pass)
# sns.countplot(x=labels, data = studentData, palette='hls')
# plt.savefig('countplot')

# table=pd.crosstab(studentData.gender, studentData.final_result)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Gender vs Final Result')
# plt.xlabel('Gender')
# plt.ylabel('Final Result')
# plt.savefig('gender_grade')
#
# table=pd.crosstab(studentData.disability, studentData.final_result)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of disability vs Final Result')
# plt.xlabel('Disability')
# plt.ylabel('Final Result')
# plt.savefig('disability_grade')
#
# table=pd.crosstab(studentData.highest_education, studentData.final_result)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Highest Education vs Final Result')
# plt.xlabel('Highest Education')
# plt.ylabel('Final Result')
# plt.savefig('highested_grade')
#
# table=pd.crosstab(studentData.region, studentData.final_result)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Region vs Final Result')
# plt.xlabel('Region')
# plt.ylabel('Final Result')
# plt.savefig('region_grade')
#
# table=pd.crosstab(studentData.age_band, studentData.final_result)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Age Band vs Final Result')
# plt.xlabel('Age Band')
# plt.ylabel('Final Result')
# plt.savefig('agebands_grade')
#
# table=pd.crosstab(studentData.imd_band, studentData.final_result)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of imd band vs Final Result')
# plt.xlabel('IMD Band')
# plt.ylabel('Final Result')
# plt.savefig('imdband_grade')
#
# table=pd.crosstab(studentData.num_of_prev_attempts, studentData.final_result)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Previous Attempts vs Final Result')
# plt.xlabel('Previous attempts')
# plt.ylabel('Final Result')
# plt.savefig('prevattempts_grade')

# table=pd.crosstab(studentData.code_module, studentData.final_result)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Module Code vs Final Result')
# plt.xlabel('Module code')
# plt.ylabel('Final Result')
# plt.savefig('codemodule_grade')
#
# table=pd.crosstab(studentData.code_presentation, studentData.final_result)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Presentation code vs Final Result')
# plt.xlabel('Presentation code')
# plt.ylabel('Final Result')
# plt.savefig('codepresentation_grade')

# table=pd.crosstab(studentData.studied_credits, studentData.final_result)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Studied Credits vs Final Result')
# plt.xlabel('Studied credits')
# plt.ylabel('Final Result')
# plt.savefig('studiedcredits_grade')

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets, 75% training, 25% test
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25)

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 100 decision trees, of a max depth 3
# Bootstrap refers to sample being randomly selected
rf = RandomForestClassifier(n_estimators = 100, max_depth=3, bootstrap = True)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
rfPredictions = rf.predict(test_features)
# Printing first 10 test labels and predictions
print('Test labels: ')
print(test_labels[:10])
print('Predictions: ' )
print(rfPredictions[:10])


# Probabilities for each class
rf_probs = rf.predict_proba(test_features)[:, 1]

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_labels, rfPredictions))

# Printing classification report with recall and precision data
from sklearn.metrics import classification_report
print('Random Forest Classification: Test Report')
print(classification_report(test_labels,rfPredictions))

#Printing confusion matrix
from sklearn.metrics import confusion_matrix
print('Random Forest Classification: Confusion Matrix')
print(confusion_matrix(test_labels, rfPredictions))

# Saving a random forest tree to rfsampletree.png
from sklearn.tree import export_graphviz
import pydot
tree_small = rf.estimators_[5]
export_graphviz(tree_small, out_file = 'rfsampletree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('rfsampletree.dot')
graph.write_png('rfsampletree.png');

# Making ROC curve
from sklearn.metrics import roc_curve, auc

Y_score = rf.predict_proba(test_features)[:,1]
fpr = dict()
tpr = dict()
fpr, tpr, _ = roc_curve(test_labels, Y_score)
roc_auc = dict()
roc_auc = auc(fpr, tpr)

# make the plot
plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.plot(fpr, tpr, label='AUC = {0}'.format(roc_auc))
plt.legend(loc="lower right", shadow=True, fancybox =True)
plt.show()

# ---------------- end of random forest classification -------------------
# ---------------- LOGISTIC REGRESSION -----------------------------------

from sklearn.linear_model import LogisticRegression

# Separating training and test data 70% and 25% respectively
X_train, X_test, y_train, y_test = train_test_split(features,
           labels, test_size = 0.25)

# Initialising the logistic regression model
logmodel = LogisticRegression(max_iter=1000)
# Fitting the logistic regression model on the training data
logmodel.fit(X_train, y_train)


# Predicting on the test data
logPredictions = logmodel.predict(X_test)
# Printing first 10 test label data and what they were predicted
print('Test labels:')
print(y_test[:10])
print('Test predictions:')
print(logPredictions[:10])

# Printing classification data - including precision and recall
print('Logistic Regression Classification: Test report')
print(classification_report(y_test, logPredictions))

# Printing confusion matrix
print('Logistic Regression Classification: Confusion Matrix')
print(confusion_matrix(y_test, logPredictions))

# Printing ROC curve
Y_score = rf.predict_proba(X_test)[:,1]
fpr = dict()
tpr = dict()
fpr, tpr, _ = roc_curve(y_test, Y_score)
roc_auc = dict()
roc_auc = auc(fpr, tpr)

# make the plot
plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.plot(fpr, tpr, label='AUC = {0}'.format(roc_auc))
plt.legend(loc="lower right", shadow=True, fancybox =True)
plt.show()
