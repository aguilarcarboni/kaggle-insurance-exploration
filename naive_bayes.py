# Softmax Regressor
# Jose Miguel Loguirato, Andres Aguilar, Maria Elena Leizaola
# CS 3368

import numpy as np
import pandas as pd
import pandas as pd
from sklearn import metrics, model_selection, metrics

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

print('Training Naive Bayes model on dataset.')

centroid_image = np.mean(train_data, axis=0)
class_list = np.unique(train_labels)
num_classes = len(class_list)
num_pixels = train_data.shape[1]
num_test_cases = len(test_labels)

# Loop through every class
prob_class_img = np.zeros( (num_classes, num_pixels) )

for class_index in range(num_classes):

    # Create an image of average pixels for this class
    mask = train_labels==class_index
    train_data_this_class = np.compress(mask, train_data, axis=0)

    class_centroid_image = np.mean(train_data_this_class, 0)

    # Compute probability of class for each pixel
    prob_class_img[class_index] = class_centroid_image / (centroid_image+.0001) / num_classes

# Now use the probability images to estimate the probability of each class
# in new images
pred = np.zeros(num_test_cases)

# Predict all test images
for text_index in range(num_test_cases):

    test_img = test_data[text_index]

    prob_class = []
    for classidx in range(num_classes):
        
        test_img_prob_class = test_img * prob_class_img[classidx]

        # Average the probabilities of all pixels
        prob_class.append( np.mean(test_img_prob_class) )

    # Pick the largest
    pred[text_index] = prob_class.index(max(prob_class))

print("ROC:   {:.3f}".format(metrics.roc_auc_score(test_labels, pred)))