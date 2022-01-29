from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score


# import data
df2 = pd.read_csv('track.csv')

# data cleaning
dfrel = df2.loc[:, ['applicant_age', 'race', 'ethnicity',
                    'sex', 'income', 'loan_amount', 'accepted', 'debt_to_income_ratio', 'combined_loan_to_value_ratio']]
dfrel.dropna(inplace=True)

# one-hot encoding
dummies = pd.get_dummies(dfrel['race'])
dummies1 = pd.get_dummies(dfrel['sex'])
dummies2 = pd.get_dummies(dfrel['applicant_age'])
dummies3 = pd.get_dummies(dfrel['ethnicity'])
dummies4 = pd.get_dummies(dfrel['debt_to_income_ratio'])
dfrel = pd.concat([dfrel, dummies, dummies1, dummies2,
                   dummies3, dummies4], axis=1)
dfrel.drop(labels=["race", "applicant_age", "sex",
                   "ethnicity", "debt_to_income_ratio", ], axis=1, inplace=True)


# encode binary variable
def encode_binary_variable(df, col, val_pos, val_neg):
    df[col] = df[col].replace({val_pos: 1, val_neg: 0})


encode_binary_variable(dfrel, "accepted", 1.0, 0.0)

mms = MinMaxScaler()
dfrel = pd.DataFrame(mms.fit_transform(dfrel.values), columns=dfrel.columns)


# training
X = dfrel.drop(labels=["accepted"], axis=1)
y = dfrel["accepted"]


# original data, accepted/not accepted bar graph
ax = df2['accepted'].value_counts().plot(
    kind='bar', figsize=(10, 6), fontsize=13, color='#087E8B')
ax.set_title('Mortgage Accepted (0 = not accepted, 1 = accepted)',
             size=20, pad=30)
ax.set_ylabel('Number of mortgages accepted ', fontsize=14)

for i in ax.patches:
    ax.text(i.get_x() + 0.19, i.get_height() + 700,
            str(round(i.get_height(), 2)), fontsize=15)


# create testing dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=0)


# logistic regresiion
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


# utilize SMOTE to fix unbalanced data classes
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_val, y_val)
print(f'''Shape of X before SMOTE: {X_val.shape}
Shape of X after SMOTE: {X_sm.shape}''')

print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize=True) * 100

X_train, X_test, y_train, y_test = train_test_split(
    X_sm, y_sm, test_size=0.25, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(
    f'Accuracy = {accuracy_score(y_test, preds):.2f}\nRecall = {recall_score(y_test, preds):.2f}\n')
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix (with SMOTE)', size=16)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.savefig('heatmap.png')


print("Accuracy SMOTE: ", accuracy_score(y_val, y_pred_sm))
# precision and recall accuracy for accepted
print("Precision for Accepted Applications SMOTE: ",
      precision_score(y_val, y_pred_sm))
print("Recall for Accepted Applications SMOTE: ", recall_score(y_val, y_pred_sm))
# precision and recall accuracy for rejected
print("Precision for Rejected Applications SMOTE: ",
      precision_score(y_val, y_pred_sm, pos_label=0))
print("Recall for Rejected Applications SMOTE: ",
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


print("F1 Score for the Baseline Model (Accepted): ", f1_score(y_val, y_pred))
print("F1 Score for the SMOTE Model (Accepted): ", f1_score(y_val, y_pred_sm))
print("F1 Score for the Baseline Model (Rejected): ",
      f1_score(y_val, y_pred, pos_label=0))
print("F1 Score for the SMOTE Model (Rejected): ",
      f1_score(y_val, y_pred_sm, pos_label=0))


dftest = dfrel.copy()

dftest.loc[:, 'Native Hawaiian or Other Pacific Islander'] = 0.0
dftest.loc[:, 'White'] = 1.0
dftest.loc[:, 'Asian'] = 0.0
dftest.loc[:, 'Black or African American'] = 0.0
dftest.loc[:, 'American Indian or Alaska Native'] = 0.0
dftest.loc[:, '2 or more minority races'] = 0.0
dftest.loc[:, 'Free Form Text Only'] = 0.0


# compare predicted approval rates if race is changed to white
X = dfrel.drop(labels=["accepted"], axis=1)
X2 = dftest.drop(labels=["accepted"], axis=1)
white_preds = model.predict(X2)
preds2 = model.predict(X)

dfrel["preds"] = preds2
dfrel["white_preds"] = white_preds
dfnonwhite = dfrel[dfrel.White == 0.0]
print(dfnonwhite.head(10))
black_orig = dfnonwhite[dfnonwhite["Black or African American"] ==
                        1.0]["preds"].mean()
black_skew = dfnonwhite[dfnonwhite["Black or African American"] ==
                        1.0]["white_preds"].mean()
asian_orig = dfnonwhite[dfnonwhite["Asian"] == 1.0]["preds"].mean()
asian_skew = dfnonwhite[dfnonwhite["Asian"] ==
                        1.0]["white_preds"].mean()
na_orig = dfnonwhite[dfnonwhite["American Indian or Alaska Native"] ==
                     1.0]["preds"].mean()
na_skew = dfnonwhite[dfnonwhite["American Indian or Alaska Native"] ==
                     1.0]["white_preds"].mean()
hawaii_orig = dfnonwhite[dfnonwhite["Native Hawaiian or Other Pacific Islander"] ==
                         1.0]["preds"].mean()
hawaii_skew = dfnonwhite[dfnonwhite["Native Hawaiian or Other Pacific Islander"] ==
                         1.0]["white_preds"].mean()

# logistic regression graph
#sns.regplot(x=X_test, y=preds, data=dfrel, logistic=True, ci=None)

race_labels = ['Black', 'Asian',
               'Native American', 'Hawaiian/Pacific Islander']
original_means = [black_orig, asian_orig, na_orig, hawaii_orig]
skewed_means = [black_skew, asian_skew, na_skew, hawaii_skew]

# label locations
x_labels = np.arange(len(race_labels))
# width of bars
width_bar = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x_labels - width_bar/2, original_means,
                width_bar, label='Original')
rects2 = ax.bar(x_labels + width_bar/2, skewed_means,
                width_bar, label='Skewed')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Approval Rate')
ax.set_xlabel('Race')
ax.set_title('Investigating Racial Impact on Mortgage Approval')
ax.set_xticks(x_labels, race_labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig("approvalratebar.png")
plt.show()
