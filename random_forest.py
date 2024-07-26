# Softmax Regressor
# Jose Miguel Loguirato, Andres Aguilar, Maria Elena Leizaola
# CS 3368

import pandas as pd
from sklearn import metrics, ensemble, model_selection

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

print('Training Random Forest Regression model on dataset.')

# Create and fit model
model = ensemble.RandomForestClassifier(n_estimators = 1, min_samples_leaf = 1e-4) 
model.fit(train_data, train_labels)

print('Predicting...')
pred = model.predict(test_data)

print("ROC:   {:.3f}".format(metrics.roc_auc_score(test_labels, pred)))