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

# Modifiable parameters
tolerance = 1e-3
max_iter = 100

# Create and fit model on data
print('Training Logistic Regression model on dataset.')
model = linear_model.LogisticRegression(solver='sag', tol=tolerance, max_iter = max_iter) 
model.fit(train_data, train_labels)

# Generate ROCAUC score
print('Predicting...')
pred = model.predict(test_data)
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
prediction = model.predict(submission_data)
print('Done!')