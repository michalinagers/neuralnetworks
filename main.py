import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential #for neural networks
from tensorflow.keras.layers import Dense #create dense neural networks
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from google.colab import drive
drive.mount('/content/drive')

diabetes_data = pd.read_csv('/content/drive/MyDrive/week12/diabetes.csv')

diabetes_data.head()

x = diabetes_data.drop('Outcome', axis=1)
#x = diabetes_data.drop(columns['Columns'])
y = diabetes_data['Outcome'] #slicing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#input layer = amount of features in ur dataset
#in this dataset it is 8, as we not using 'Outcome'
#2 hidden layers, 1 is 16 neurons and 2nd is 8 neurons, use RELU
#if binary output layer is 1 and use sigmoid, if multi use soft max foctions

input_dim = x_train.shape[1]

#creating layers, order in which u write matters
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

input_dim
x.shape

model.summary()
plot_model(model, to_file='model_plot.jpg', show_shapes=True, show_layer_names=True)

model.compile(loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50, verbose=1) #run it for longer to increase accuracy

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

y_pred = (model.predict(x_test) > 0.5).astype(int)
#y_pred = (y_pred > 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

test_data = [2, 197, 70, 45, 543, 30.5, 0.158, 53] #testdata

test_data = np.array(test_data).reshape(1, -1)
test_data_scaled = scaler.transform(test_data)

prediction_prob = model.predict(test_data_scaled)[0][0]

print("Predicted probability: ", prediction_prob)

treshold = 0.5 #above 0.5 has diabates, below 0.5 does not
prediction = int(prediction_prob > treshold)

print("Predicted class: ", prediction)
