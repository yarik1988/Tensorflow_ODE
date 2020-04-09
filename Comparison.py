import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy import integrate

t_init=0.0
t_final=10.0
y0=-0.1
y_init = tf.constant([y0], dtype=tf.float64)

def ode_fn(t, y): return tf.math.sin(t)/(t+y+1)


print("DormandPrince...")
results_dp = tfp.math.ode.DormandPrince(safety_factor=0.02).solve(ode_fn, t_init, y_init,solution_times=tfp.math.ode.ChosenBySolver(final_time=t_final))
print("BDF...")
results_bdf = tfp.math.ode.BDF(safety_factor=0.8).solve(ode_fn, t_init, y_init,solution_times=tfp.math.ode.ChosenBySolver(final_time=t_final))
print("Scipy")
t=np.linspace(t_init, t_final, 1000)
sol = integrate.odeint(lambda y,t:ode_fn(t,y),y0,t)
plt.plot(results_dp.times,results_dp.states,'r', label='DormandPrince')
plt.plot(results_bdf.times,results_bdf.states,'b', label='BDF')
plt.plot(t, sol,'g', label='Scipy')
plt.legend(loc='best')
plt.title("BDF: {} points, DP: {} points".format(len(results_bdf.times),len(results_dp.times)))
plt.grid()
plt.show()
