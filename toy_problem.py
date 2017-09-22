import isochronism
import numpy as np
import tensorflow as tf

n_freq = 10
n_params = 3
n_batch = 20
batch_size = 5
step_size = 1e-2
n_training_steps = 10
f0 = 10
I = 1. / 2. / np.pi**2
ampl = 0.6*np.linspace(1, 10, n_freq)


def make_data(fname, ampl_max):
    n = 50
    a0 = 0.5
    a1 = 1
    b0 = 0.5
    c0 = 0.5
    data = np.empty((n, 2))
    data[:,0] = np.linspace(-ampl_max, ampl_max, n)
    a = a1 + a0*2*(np.random.rand()-0.5)
    b = b0*2*(np.random.rand()-0.5)
    c = c0*2*(np.random.rand()-0.5)
    data[:,1] = a*data[:,0] + b*np.power(data[:,0], 2) + c*np.power(data[:,0], 3)
    np.savetxt(fname, data)
    return [a, b, c]

def data_batch(size):
    ampl_max = 6
    fname = 'temp.txt'
    params = np.empty((size, n_params))
    freqs = np.empty((size, n_freq))
    for i in range(size):
        params[i,:] = make_data(fname, ampl_max)
        bs = isochronism.BalanceSpring(I, fname)
        freqs[i,:] = bs.frequency(ampl)
    return [params, freqs]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


sess = tf.InteractiveSession()

parameters = tf.placeholder(tf.float32, shape=[None, n_params])
frequencies = tf.placeholder(tf.float32, shape=[None, n_freq])

nFC = 3
size1 = 10

W_fcN = [None]*nFC
b_fcN = [None]*nFC
h_fcN = [None]*nFC

W_fcN[0] = weight_variable([n_params, size1] )
b_fcN[0] = bias_variable([size1] )
h_fcN[0] = tf.matmul(parameters, W_fcN[0]) + b_fcN[0]

for i in range(1, nFC):
    W_fcN[i] = weight_variable([size1, size1] )
    b_fcN[i] = bias_variable([size1] )
    h_fcN[i] = tf.nn.tanh(tf.matmul(h_fcN[i-1], W_fcN[i]) + b_fcN[i])

W_fc1 = weight_variable([size1, n_freq] )
b_fc1 = bias_variable([n_freq] )
freq_estimate = tf.nn.tanh(tf.matmul(h_fcN[nFC-1], W_fc1) + b_fc1)
loss = tf.losses.mean_squared_error(frequencies, freq_estimate) / (n_batch*n_freq)
train_step = tf.train.AdamOptimizer(step_size).minimize(loss)
sess.run(tf.global_variables_initializer())

for i in range(n_training_steps):
    params, freqs = data_batch(n_batch)
    train_step.run(feed_dict={parameters: params, frequencies: freqs})
    a = loss.eval(feed_dict={parameters: params, frequencies: freqs})
    print('%i -- %f'%(i,a))
    #print(params)
    #print(freqs)


params = [[1, 0, 0]]
freqs = freq_estimate.eval(feed_dict={parameters: params})
print(freqs)
