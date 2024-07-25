import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.model_selection
import time
import math
import matplotlib.pyplot as plt

# Load and prepare the data
df = pd.read_csv("data/train.csv", header=0)

# Preprocess data
df = pd.get_dummies(df, prefix_sep="_", drop_first=True, dtype=int)

labels = df["Response"]
df = df.drop(columns="Response")

# Split data
train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(
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
    
# Get some properties of data
num_inputs = train_data.shape[1]
num_samples = train_data.shape[0]

# Training constants
num_nodes = 76
num_layers = 2
num_nodes_per_layer = num_nodes/num_layers
num_outputs = 1

batch_size = int(num_samples / 10)
# 1. /10
# 2. /100

n_batches = math.ceil(num_samples / batch_size)

learning_rate = 1e-3
# 1. 1e-3
# 2. 5-e3

# Linear Regression
# Works best for predicting a category from others
activation = 'sigmoid'
loss = 'binary_crossentropy'
metrics = ['AUC']

n_epochs = 1
eval_step = 1

# Create and train model
# Create a neural network model
model = tf.keras.models.Sequential()

# Add an Input layer
model.add(tf.keras.layers.Input(shape=(num_inputs,)))

# Hidden layer 1:
#   Inputs: num_inputs - columns
#   Outputs: num_nodes - arbitrary
#   Activation: ELU
for layer in range(num_layers):
        model.add(tf.keras.layers.Dense(
                        int(num_nodes_per_layer),
                        activation='elu',
                        kernel_initializer='he_normal', 
                        bias_initializer='zeros'
                )
        )

# Output layer:
#   Inputs: num_nodes
#   Outputs: num_outputs
#   Activation: none/linear

model.add(tf.keras.layers.Dense(
                num_outputs,
                activation=activation,
                kernel_initializer='glorot_normal', 
                bias_initializer='zeros'
        )
)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Define loss function for expected result
model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
)

# Train the neural network
start_time = time.time()
history = model.fit(
        train_data,
        train_labels,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(test_data, test_labels),
        validation_freq=eval_step,
        verbose=2,
)
elapsed_time = time.time() - start_time
print("Execution time: {:.1f}".format(elapsed_time))

cost_test, auc_test = model.evaluate(test_data, test_labels, batch_size=None, verbose=0)
cost_train, auc_train = model.evaluate(train_data, train_labels, batch_size=None, verbose=0)

print("Final Test AUC:          {:.4f}".format(auc_test))
print("Final Training Cost:     {:.4f}".format(cost_train))


#
predictions = []
prediction = model.predict(test_data)
for index, data in enumerate(test_data.index):
        predictions.append([data, prediction[index][0]])

df = pd.DataFrame(predictions, columns=["Id", "Response"])
print(df)
df.to_csv("data/predictions.csv", index=False)

# Compute the best test result from the history
epoch_hist = [i for i in range(0, n_epochs, eval_step)]
test_auc_hist = history.history['val_AUC']
test_best_val = max(test_auc_hist)
test_best_idx = test_auc_hist.index(test_best_val)
print("Best Test AUC:           {:.4f} at epoch: {}".format(test_best_val, epoch_hist[test_best_idx]))

# Plot the history of the loss
plt.plot(history.history['loss'])
plt.title('Training Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')

# Plot the history of the test accuracy
plt.figure()
plt.plot(epoch_hist, history.history['val_AUC'], "r")
plt.title('Test AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.show()