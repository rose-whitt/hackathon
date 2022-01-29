from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

# import data
df2 = pd.read_csv('track.csv')

# data cleaning
dfrel = df2.loc[:, ['applicant_age', 'race', 'ethnicity',
                    'sex', 'income', 'loan_amount', 'accepted', 'debt_to_income_ratio']]
dfrel.dropna(inplace=True)

dummies = pd.get_dummies(dfrel['race'])
dummies1 = pd.get_dummies(dfrel['sex'])
dummies2 = pd.get_dummies(dfrel['applicant_age'])
dummies3 = pd.get_dummies(dfrel['ethnicity'])
dummies4 = pd.get_dummies(dfrel['debt_to_income_ratio'])

dfrel = pd.concat([dfrel, dummies, dummies1, dummies2,
                   dummies3, dummies4], axis=1)
dfrel.drop(labels=["race", "applicant_age", "sex",
                   "ethnicity", "debt_to_income_ratio"], axis=1, inplace=True)


def encode_binary_variable(df, col, val_pos, val_neg):
    df[col] = df[col].replace({val_pos: 1, val_neg: 0})


encode_binary_variable(dfrel, "accepted", 1.0, 0.0)

X = dfrel.drop(labels=["accepted"], axis=1)
y = dfrel["accepted"]

# create testing dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)
logreg = LogisticRegression(random_state=1, max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_val)


dfrel['accepted'].value_counts()
print(str(dfrel['accepted'].value_counts()[1]/(dfrel['accepted'].value_counts()[1] +
                                               dfrel['accepted'].value_counts()[0])*100)+"% of entries correspond to accepted loans")
print("Accuracy: ", accuracy_score(y_val, y_pred))
# precision and recall accuracy for accepted
print("Precision for Accepted Applications: ", precision_score(y_val, y_pred))
print("Recall for Accepted Applications: ", recall_score(y_val, y_pred))
# precision and recall accuracy for rejected
print("Precision for Rejected Applications: ",
      precision_score(y_val, y_pred, pos_label=0))
print("Recall for Rejected Applications: ",
      recall_score(y_val, y_pred, pos_label=0))

smote = SMOTE(sampling_strategy='minority')
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
y_train_res.value_counts()
logreg_sm = LogisticRegression(random_state=0, max_iter=1000)
logreg_sm.fit(X_train_res, y_train_res)
y_pred_sm = logreg_sm.predict(X_val)
print("Accuracy: ", accuracy_score(y_val, y_pred_sm))
# precision and recall accuracy for accepted
print("Precision for Accepted Applications: ",
      precision_score(y_val, y_pred_sm))
print("Recall for Accepted Applications: ", recall_score(y_val, y_pred_sm))
# precision and recall accuracy for rejected
print("Precision for Rejected Applications: ",
      precision_score(y_val, y_pred_sm, pos_label=0))
print("Recall for Rejected Applications: ",
      recall_score(y_val, y_pred_sm, pos_label=0))


# ROC curve
y_pred_prob = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
auc = metrics.roc_auc_score(y_test, y_pred_prob)

plt.plot(fpr, tpr, label="AUC" + str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
