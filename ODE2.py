from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt
import keyboard
from scipy.integrate import solve_ivp
tf.keras.backend.set_floatx('float64')


ready_to_exit = False


def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit=True


def train_step(model, optimizer, x):
    N=x.shape[0]
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        with tf.GradientTape() as gg:
            gg.watch(x)
            y=model(x)
            dy = gg.gradient(y, x)
        ddy=g.gradient(dy, x)
        residual=ddy-tf.math.square(y)+y+1
        loss = tf.math.square(residual)
        start_loss = (tf.math.square(y[0])+tf.math.square(dy[0]-1))*N
        start_loss=tf.concat([start_loss, tf.zeros(N-1,dtype='float64')], 0)
        total_loss = loss+start_loss
        gradients = g.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return float(tf.math.reduce_mean(loss)),float(tf.math.reduce_mean(start_loss))


N = 500
T=5
gx = np.linspace(0,T,N)

model = keras.models.Sequential()
model.add(layers.InputLayer(input_shape=(1,)))
for k in range(5):
    model.add(layers.Dense(50, activation='elu'))
model.add(layers.Dense(1, use_bias = False))

lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.001,decay_steps=100000,decay_rate=0.99)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

bar = tqdm()
keyboard.on_press_key("q", press_key_exit)
while True:
    if ready_to_exit:
        break
    grid_tf_run=tf.expand_dims(gx,axis=1)
    loss = train_step(model, optimizer, grid_tf_run)
    bar.set_description("Loss: {:.5f} {:.5f}".format(*loss))
    bar.update(1)

#solve by Runge kutta
def diff(t, z):
    return [z[1],z[0]*z[0]-z[0]-1]
sol = solve_ivp(diff,[0, T], [0,1],dense_output=True)
z = sol.sol(gx)
y = model(tf.expand_dims(q,axis=1))
plt.plot(gx,np.squeeze(y))
plt.plot(gx, z[0].T)
plt.title('Solving equation $y\'\'=y^2-y-1$')
plt.legend(['Tensorflow','Runge-Kutta'])
plt.show()
