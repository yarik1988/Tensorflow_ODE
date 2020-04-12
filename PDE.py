from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import keyboard


ready_to_exit = False


def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit=True


def train_step(model, optimizer, features, rhs, initial_cond):
    N=int(tf.size(features))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(features)
        predictions = model(features)
        dy = tape.gradient(predictions, features)
        loss = tf.math.square(dy - rhs(x, predictions))
        v1 = tf.convert_to_tensor(np.ones(N) * initial_cond, dtype=tf.float32)
        kkk = tf.math.square(tf.squeeze(predictions) - v1)
        v2 = np.zeros(N)
        v2[0] = N
        v2 = tf.convert_to_tensor(v2, dtype=tf.float32)

        start_loss = tf.math.multiply(kkk, v2)
        start_loss = tf.expand_dims(start_loss, axis=1)
        total_loss = loss+start_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    fun_amplitude = tf.math.reduce_mean(tf.math.abs(predictions))
    return tf.math.reduce_mean(total_loss)/fun_amplitude


N = 10
lm=np.linspace(0, 1, N)
gx,gy = np.meshgrid(lm,lm)
grid = np.vstack((np.ndarray.flatten(gx),np.ndarray.flatten(gy))).transpose()
x = tf.convert_to_tensor(grid, dtype=tf.float32)
print(x.shape)

model = keras.models.Sequential()
model.add(layers.InputLayer(input_shape=(2,)))
for k in range(5):
    model.add(layers.Dense(50, activation='tanh'))
model.add(layers.Dense(1, use_bias = False))
optimizer = tf.keras.optimizers.RMSprop()
initial_cond = 0

'''
bar = tqdm()
keyboard.on_press_key("q", press_key_exit)
while True:
    if ready_to_exit:
        break
    loss = train_step(model, optimizer, x, lambda x, y: tf.math.sin(x)/(x+y+1), initial_cond)
    bar.set_description("Loss: {}".format(float(loss)))
    bar.update(1)
'''
val = model(x)
val = np.reshape(val,(N,N))

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(val,gx,gy,cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()

