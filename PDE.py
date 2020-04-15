from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import keyboard
tf.keras.backend.set_floatx('float64')

ready_to_exit = False


def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit=True


def train_step(model, optimizer, features, init_cond, init_mask):
    N=int(tf.math.reduce_sum(init_mask))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(features)
        predictions = model(features)
        dy = tape.gradient(predictions, features)
        coeff=tf.convert_to_tensor([1.0,-2.0],dtype=np.float64)
        cy=tf.math.multiply(features,coeff)
        lhs=tf.math.reduce_sum(tf.math.multiply(dy,cy), axis=1)
        rhs=tf.math.reduce_sum(tf.math.square(features),axis=1)
        loss = tf.math.square(lhs - rhs)
        kkk = tf.math.square(tf.squeeze(predictions) - init_cond)
        start_loss = tf.multiply(kkk, init_mask)
        start_loss = tf.expand_dims(start_loss, axis=1)*N
        total_loss = loss+start_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return float(tf.math.reduce_mean(loss)),float(tf.math.reduce_mean(start_loss))


N = 100
x=np.linspace(-1, 1, N)
func = lambda x,y: x*x/2 - y*y/4 + y*x*x/2 + 1/4
ground_truth = func(x[:,None], x[None,:])
gx,gy = np.meshgrid(x,x)
init_cond=np.zeros((N,N))
init_cond[:,-1]=x**2
init_cond = tf.convert_to_tensor(np.ndarray.flatten(init_cond))
init_mask=np.zeros((N,N))
init_mask[:,-1]=1
init_mask = tf.convert_to_tensor(np.ndarray.flatten(init_mask))
grid_tf = np.vstack((np.ndarray.flatten(gy),np.ndarray.flatten(gx))).transpose()
grid_tf = tf.convert_to_tensor(grid_tf)

model = keras.models.Sequential()
model.add(layers.InputLayer(input_shape=(2,)))
for k in range(1):
    model.add(layers.Dense(50, activation='elu'))
model.add(layers.Dense(1, use_bias = False))
optimizer = tf.keras.optimizers.Adam()
initial_cond = 0

bar = tqdm()
keyboard.on_press_key("q", press_key_exit)
while True:
    if ready_to_exit:
        break
    indicies = tf.random.shuffle(tf.range(N*N))
    grid_tf_run=tf.gather(grid_tf, indicies,axis=0)
    init_cond_run=tf.gather(init_cond, indicies)
    init_mask_run=tf.gather(init_mask, indicies)
    loss = train_step(model, optimizer, grid_tf_run, init_cond_run, init_mask_run)
    bar.set_description("Loss: {:.5f} {:.5f}".format(*loss))
    bar.update(1)




with tf.GradientTape() as tape:
        tape.watch(grid_tf)
        val = model(grid_tf)
        grad = np.array(tape.gradient(val,grid_tf))


val = np.reshape(val,(N,N))
step=np.mean(np.diff(x))
res=np.gradient(val,step)
xm=np.tile(x, (N,1)).transpose()
ym=np.tile(x, (N,1))
dx=np.reshape(grad[:,0],(N,N))
dy=np.reshape(grad[:,1],(N,N))
err=dx*xm-2*dy*ym-xm*xm-ym*ym
print("Mean error of differential equation: (analytical diff){}".format(np.mean(err)))
err=res[0]*xm-2*res[1]*ym-xm*xm-ym*ym
print("Mean error of differential equation (numerical diff): {}".format(np.mean(err)))
err=np.abs(val-ground_truth)
print("Difference from ground truth function: {}".format(np.mean(err)))




fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(gx,gy,val,cmap='viridis', edgecolor='none')
ax.plot_surface(gx,gy,ground_truth, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()




