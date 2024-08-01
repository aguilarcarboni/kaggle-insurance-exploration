# Gradient Boosted Trees Classifier
# Jose Miguel Loguirato, Andres Aguilar, Maria Elena Leizaola
# CS 3368

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, ensemble, metrics

# Load and prepare the data
df = pd.read_csv("data/train.csv", header=0)

# Preprocess data
df = pd.get_dummies(df, prefix_sep="_", drop_first=True, dtype=int)
labels = df["Response"]
df = df.drop(columns="Response")

# Split data
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(
        df, 
        labels,
        test_size=0.2, 
        shuffle=True, 
        random_state=2024
)

# Standadrize scale for all columns
for col in train_data.columns:
    mean = train_data[col].mean()
    stddev = train_data[col].std()
    train_data[col] = train_data[col] - mean
    train_data[col] = train_data[col]/stddev
    test_data[col] = test_data[col] - mean
    test_data[col] = test_data[col]/stddev

# Modifiable parameters
num_estimators = 10
msl = 0.001
subsampling=0.2

# Create and fit model
print('Training Gradient Boosted Trees model on dataset.')
model = ensemble.GradientBoostingClassifier(
    loss='log_loss', 
    subsample=subsampling,
    n_estimators=num_estimators,
    min_samples_leaf=msl
)
model.fit(train_data, train_labels)

# Generate ROCAUC score
pred_proba = model.predict_proba(test_data)[:,1]
auc_score = metrics.roc_auc_score(test_labels, pred_proba)
print("Test AUC score: {:.4f}".format(auc_score))

# Compute a precision & recall graph
precisions, recalls, thresholds = metrics.precision_recall_curve(test_labels, pred_proba)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend(loc="center left")
plt.xlabel("Threshold")
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.show()

# Plot a ROC curve
fpr, tpr, _ = metrics.roc_curve(test_labels, pred_proba)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()

# Read submission data
df_data = pd.read_csv('data/test.csv')
submission_data = pd.get_dummies(df_data, prefix_sep="_", drop_first=True, dtype=int)

# Standadrize scale for all columns
for col in submission_data.columns:
    mean = submission_data[col].mean()
    stddev = submission_data[col].std()
    submission_data[col] = submission_data[col] - mean
    submission_data[col] = submission_data[col]/stddev

# Predict data
print('Creating submission data...')
prediction = model.predict(submission_data)
print('Done!')