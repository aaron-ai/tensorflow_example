import tensorflow as tf
import numpy as np

X_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([5.0, 8.0, 11.0, 14.0])

# define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

# compile the model, loss function and optimizer.
model.compile(optimizer='sgd', loss='mean_squared_error')

# train
model.fit(X_train, y_train, epochs=500)

# predit
X_test = np.array([5.0, 6.0, 7.0])
y_pred = model.predict(X_test)

print(y_pred)
