import tensorflow as tf
import numpy as np

X_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([5.0, 8.0, 11.0, 14.0])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

optimizer = tf.keras.optimizers.SGD()
loss_fn = tf.keras.losses.MeanSquaredError()

# customize training loop
for epoch in range(500):
    with tf.GradientTape() as tape:
        y_pred = model(X_train, training=True)  # Forward pass
        loss = loss_fn(y_train, y_pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

X_test = np.array([5.0, 6.0, 7.0])
# predict
y_pred = model.predict(X_test)

print(y_pred)
