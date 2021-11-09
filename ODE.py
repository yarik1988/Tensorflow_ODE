from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import keyboard
from scipy.integrate import solve_ivp
import models
tf.keras.backend.set_floatx('float64')
ready_to_exit = False
N = 500  # number of points to eval
diff_model = models.TestModel3
gx = np.linspace(diff_model.T[0], diff_model.T[1], N)


def get_pos_ind(i, N):
    return round((diff_model.pos[i] - diff_model.T[0]) / (diff_model.T[1] - diff_model.T[0]) * (N - 1))

def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit = True


def train_step(model, optimizer, x):
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        y = model(x)
        if diff_model.dim>1:
            dy = tf.stack([g.gradient(y[:,i], x) for i in range(diff_model.dim)],axis=1)
            dy = tf.squeeze(dy,axis=2)
            rhs = tf.stack(diff_model.tf_equation(x, y), axis=1)
        else:
            dy = g.gradient(y, x)
            rhs = diff_model.tf_equation(x, y)

        residual = dy-rhs
        loss = tf.math.square(residual)
        ind_list=[]
        val=[]
        for i in range(diff_model.dim):
            ind=get_pos_ind(i,N)
            ind_list.append([ind, i])
            ysl = y[ind, i]
            val.append((tf.math.square(ysl - diff_model.val[i])) * N)
        start_loss = tf.sparse.SparseTensor(indices=ind_list, values=val, dense_shape=loss.shape)
        start_loss = tf.sparse.to_dense(start_loss)
        total_loss = loss+start_loss
        gradients = g.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return float(tf.math.reduce_mean(loss)), float(tf.math.reduce_mean(start_loss))


model = keras.models.Sequential()
model.add(layers.InputLayer(input_shape=(1,)))
for k in range(5):
    model.add(layers.Dense(50, activation='sigmoid'))
model.add(layers.Dense(len(diff_model.pos)))
lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=100, decay_rate=0.99)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

bar = tqdm()
keyboard.on_press_key("q", press_key_exit)
min_loss = np.Inf
best_model = tf.keras.models.clone_model(model)
while True:
    if ready_to_exit:
        break
    rand_pts = np.random.uniform(low=diff_model.T[0], high=diff_model.T[1], size=(N,))
    rand_pts = np.sort(rand_pts)
    for i in range(diff_model.dim):
        rand_pts[get_pos_ind(i, N)] = diff_model.pos[i]
    grid_tf_run = tf.expand_dims(rand_pts, axis=1)
    loss = train_step(model, optimizer, grid_tf_run)
    if sum(loss) < min_loss:
        min_loss = sum(loss)
        best_model.set_weights(model.get_weights())
    bar.set_description("Loss: {:.5f} {:.5f} min loss: {:.6f}".format(*loss, min_loss))
    bar.update(1)

y = best_model(tf.expand_dims(gx,axis=1))
plt.plot(gx, np.squeeze(y),'m-', label='Tensorflow')
if np.all(diff_model.pos == diff_model.T[0]):
    sol = solve_ivp(diff_model.equation, diff_model.T, diff_model.val, dense_output=True)
    rk = sol.sol(gx)
    plt.plot(gx, rk.T, 'g-', label='Runge-Kutta')
if hasattr(diff_model, 'theoretical') and callable(getattr(diff_model, 'theoretical')):
    plt.plot(gx, diff_model.theoretical(gx),'b-',label='Theoretical')
plt.title(diff_model.__doc__)
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc='best')
plt.show()
