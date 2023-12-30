import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
# Import dataset
df = pd.read_csv("framingham.csv")

# Data Cleaning
df.education.fillna(0, inplace=True)
df.cigsPerDay.fillna(df.cigsPerDay.where(df.currentSmoker == 1).median(), inplace=True)
df.BPMeds.fillna(0, inplace=True)
df['totChol'].fillna(df.totChol.median(), inplace=True)
df['BMI'].fillna(df.BMI.median(), inplace=True)
df['heartRate'].fillna(df['heartRate'].where(df['currentSmoker'] == 1).median(), inplace=True)
df['glucose'].fillna(df['glucose'].where(df['diabetes'] == 0).median(), inplace=True)

# List of columns names with contineous values
col = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
# Making a copy of the clean dataframe
df1 = df.copy()

# To remove outliers
for i in col:
    q1 = df1[i].quantile(q=0.25)
    q2 = df1[i].quantile()
    q3 = df1[i].quantile(q=0.75)
    iqr = q3 - q1
    ul = q3 + 1.5 * iqr
    ll = q1 - 1.5 * iqr

    df1 = df1[(df1[i] < ul) & (df1[i] > ll)]

X = df1.drop(['TenYearCHD'], axis=1)
y = df1['TenYearCHD']

# Keeping the significant variables
X_sig = df1[['male', 'age', 'education', 'cigsPerDay', 'prevalentStroke', 'prevalentHyp', 'sysBP', 'diaBP', 'BMI',
             'heartRate', 'glucose']]
# Standardization due to variation in scales:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_sig = scaler.fit_transform(X_sig)
# Splitting the data into train and test with significant features:
X_train, X_test, y_train, y_test = train_test_split(X_sig, y, test_size=0.30, random_state=1)

# Logistic Regression:
logreg = LogisticRegression(solver='liblinear', fit_intercept=True)

logreg.fit(X_train, y_train)

y_prob = logreg.predict_proba(X_test)[:, 1]
y_pred = logreg.predict(X_test)
y_pred2 = logreg.predict(X_train)
print('\nConfusion Matrix - Test: ', '\n', confusion_matrix(y_test, y_pred))
print('\nOverall accuracy - Test: ', '\n', accuracy_score(y_test, y_pred))
print('\nClassification report for test:\n', classification_report(y_test, y_pred))

print('\nConfusion Matrix - Test: ', '\n', confusion_matrix(y_test, y_pred2))
print('\nOverall accuracy - Test: ', '\n', accuracy_score(y_test, y_pred2))
print('\nClassification report for test:\n', classification_report(y_test, y_pred2))
# Over-sampling using SMOTE:
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=1)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
logreg_sm = LogisticRegression(solver='liblinear', fit_intercept=True)

logreg_sm.fit(X_train_sm, y_train_sm)

y_predsm = logreg_sm.predict(X_test)

print('\nConfusion Matrix - Test: ', '\n', confusion_matrix(y_test, y_predsm))
print('\nOverall accuracy - Test: ', accuracy_score(y_test, y_predsm))
print('\nClassification report for test:\n', classification_report(y_test, y_predsm))

# Confusion matrixw
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm,
                           columns=['Predicted:0', 'Predicted:1'],
                           index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sn.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")

plt.show()

cm2 = confusion_matrix(y_test, y_predsm)
conf_matrix = pd.DataFrame(data=cm2,
                           columns=['Predicted:0', 'Predicted:1'],
                           index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sn.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")

plt.show()