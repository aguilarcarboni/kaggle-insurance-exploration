# Softmax Regressor
# Jose Miguel Loguirato, Andres Aguilar, Maria Elena Leizaola
# CS 3368

import pandas as pd
from sklearn import metrics, linear_model, model_selection

# Preprocess data
df = pd.read_csv("data/train.csv", header=0)
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

# Create and fit model on data
model = linear_model.LogisticRegression(tol=5e-3, solver='newton-cg')
model.fit(train_data, train_labels)

# Generate ROCAUC score
pred = model.predict_proba(test_data)[:,1]
print("ROC:   {:.3f}".format(metrics.roc_auc_score(test_labels, pred)))

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
prediction = model.predict_proba(submission_data)[:,1]
print('Done!')