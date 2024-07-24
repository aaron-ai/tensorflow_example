import tensorflow as tf
import numpy as np

# define the input function
def input_fn():
    X_train = np.array([1.0, 2.0, 3.0, 4.0])
    y_train = np.array([5.0, 8.0, 11.0, 14.0])
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.repeat().batch(4)
    return dataset

# define the feature columns
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# create an estimator
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# train the model
estimator.train(input_fn=input_fn, steps=500)

# define the input function for prediction
def predict_input_fn():
    X_test = np.array([5.0, 6.0, 7.0])
    dataset = tf.data.Dataset.from_tensor_slices(X_test)
    dataset = dataset.batch(3)
    return dataset

# predict
predictions = estimator.predict(input_fn=predict_input_fn)
y_pred = np.array([item['predictions'][0] for item in predictions])

print(y_pred)