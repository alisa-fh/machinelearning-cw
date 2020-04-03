
#DATA PREPARATION
import pandas as pd
# Read in data and display first 5 rows
studentData = pd.read_csv('studentInfo.csv')
# Remove student entries where results are "Withdrawn" or "Distinction"
studentData = studentData[studentData.final_result != "Withdrawn"]
studentData = studentData[studentData.final_result != "Distinction"]

studentData["final_result"] = studentData["final_result"].replace("Fail", 0)
studentData["final_result"] = studentData["final_result"].replace("Pass", 1)


# One-hot encode the data using pandas get_dummies
studentDataEnc = pd.get_dummies(studentData)


# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(studentDataEnc['final_result'])

# Remove the labels from the features

# axis 1 refers to the columns
features= studentDataEnc.drop('final_result', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

#DATA VISUALISATION
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Countplot of 0 (fail) vs 1 (pass)
sns.countplot(x=labels, data = studentData, palette='hls')
plt.savefig('countplot')

# Mean of data with respect to final result
meanData = studentDataEnc.groupby('final_result').mean()
print('Mean Data:')
print(meanData.head())

# Heatmap to show where data is missing
sns.heatmap(studentData.isnull(),yticklabels=False,cbar=False,cmap='viridis')

table=pd.crosstab(studentData.gender, studentData.final_result)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Gender vs Final Result')
plt.xlabel('Gender')
plt.ylabel('Final Result')
plt.savefig('gender_grade')

table=pd.crosstab(studentData.disability, studentData.final_result)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of disability vs Final Result')
plt.xlabel('Disability')
plt.ylabel('Final Result')
plt.savefig('disability_grade')

table=pd.crosstab(studentData.highest_education, studentData.final_result)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Highest Education vs Final Result')
plt.xlabel('Highest Education')
plt.ylabel('Final Result')
plt.savefig('highested_grade')

table=pd.crosstab(studentData.region, studentData.final_result)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Region vs Final Result')
plt.xlabel('Region')
plt.ylabel('Final Result')
plt.savefig('region_grade')

table=pd.crosstab(studentData.age_band, studentData.final_result)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Age Band vs Final Result')
plt.xlabel('Age Band')
plt.ylabel('Final Result')
plt.savefig('agebands_grade')

table=pd.crosstab(studentData.imd_band, studentData.final_result)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of imd band vs Final Result')
plt.xlabel('IMD Band')
plt.ylabel('Final Result')
plt.savefig('imdband_grade')

table=pd.crosstab(studentData.num_of_prev_attempts, studentData.final_result)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Previous Attempts vs Final Result')
plt.xlabel('Previous attempts')
plt.ylabel('Final Result')
plt.savefig('prevattempts_grade')

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 10, max_depth=3, bootstrap = True)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
rfPredictions = rf.predict(test_features)

#probabilities for each class
rf_probs = rf.predict_proba(test_features)[:, 1]

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_labels, rfPredictions))

from sklearn.metrics import classification_report
print(classification_report(test_labels,rfPredictions))

from sklearn.metrics import confusion_matrix
print('confusion matrix')
print(confusion_matrix(test_labels, rfPredictions))

from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree_small = rf.estimators_[5]
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');


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



#LOGISTIC REGRESSION



from sklearn.linear_model import LogisticRegression

#create an instance and fit the model

X_train, X_test, y_train, y_test = train_test_split(features,
           labels, test_size=0.30)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

#predictions
logPredictions = logmodel.predict(X_test)

print(classification_report(y_test, logPredictions))

print('confusion matrix')
print(confusion_matrix(y_test, logPredictions))
