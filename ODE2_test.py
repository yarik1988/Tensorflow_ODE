from ODE_wrapper import WrapperODE

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
    ready_to_exit = True


def setup_ode(model):
    ode = WrapperODE(model)
    func = lambda x: ode.d(x, 2) - 6*tf.math.square(ode.y(x))
    ode.set_function(func)
    ode.set_start_values(np.asarray([1, -2]))
    return ode


def train_step(model, optimizer, ode, x):

    with tf.GradientTape(persistent=True) as g:
        g.watch(x)

        [common_loss, start_loss] = ode.loss(x)
        total_loss = common_loss + start_loss

        gradients = g.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return float(tf.math.reduce_mean(common_loss)), float(tf.math.reduce_mean(start_loss))


N = 500
T = 2
gx = np.linspace(1,T,N)


model = keras.models.Sequential()
model.add(layers.InputLayer(input_shape=(1,)))
for k in range(5):
    model.add(layers.Dense(50, activation='elu'))
model.add(layers.Dense(1, use_bias = False))

ode = setup_ode(model)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.001,decay_steps=100,decay_rate=0.99)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

bar = tqdm()
keyboard.on_press_key("q", press_key_exit)
min_loss=np.Inf
best_model=tf.keras.models.clone_model(model)
while True:
    if ready_to_exit:
        break
    grid_tf_run=tf.expand_dims(gx,axis=1)
    loss = train_step(model, optimizer, ode, grid_tf_run)
    if (sum(loss)<min_loss):
        min_loss=sum(loss)
        best_model.set_weights(model.get_weights()) 
    bar.set_description("Loss: {:.5f} {:.5f} min loss: {:.6f}".format(*loss,min_loss))
    bar.update(1)

#solve by Runge kutta
def diff(t, z):
    return [z[1], 6*z[0]*z[0]]
sol = solve_ivp(diff,[1, T], [1,-2],dense_output=True)
z = sol.sol(gx)
y = best_model(gx)
plt.plot(gx,np.squeeze(y))
plt.plot(gx, z[0].T)
plt.plot(gx, 1.0/gx**2)
plt.title('Solving equation $y\'\'=6y^2$')
plt.legend(['Tensorflow','Runge-Kutta','Theoretical'])
plt.show()