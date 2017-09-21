import isochronism

n_freq = 10
b_params = 3
batch_size = 5
n_training_steps = 10
f0 = 10
I = 1
ampl = 30*np.linspace(1, 10, n_freq)


def make_data(fname, ampl_max):
    a0 = 0.5
    a1 = 1
    b0 = 0.5
    c0 = 0.5
    x = np.linspace(-ampl, ampl, 50)
    a = a1 + a0*2*(np.random-0.5)
    b = b0*2*(np.random-0.5)
    c = c0*2*(np.random-0.5)
    data = a * np.power(x, 2) + b * np.power(x, 2) + c * np.power(x, 2)
    np.savetxt(data)
    return [a, b, c]

def data_batch(size):
    ampl_max = 300
    fname = 'temp.txt'
    params = np.empty((size, n_params))
    freqs = np.empty((size, n_freq))
    for i in range(size):
        params[i,:] = make_data(fname, ampl_max)
        bs = isochronism.BalanceSpring(I, fname)
        freqs[i,:] = bs.frequency(ampl)
    return [params, freqs]

sess = tf.InteractiveSession()

parameters = tf.placeholder(tf.float32, shape=[None, n_params]):
frequencies = tf.placeholder(tf.float32, shape=[None, n_freq]):

nFC = 2
size1 = 5
size2 = 1

W_fcN = [None]*nFC
b_fcN = [None]*nFC
h_fcN = [None]*nFC

W_fcN[0] = weight_variable([1, size1], key)
b_fcN[0] = bias_variable([size1], key)
h_fcN[0] = tf.matmul(parameters, W_fcN[0]) + b_fcN[0]

for i in range(1, nFC):
    W_fcN[i] = weight_variable([size1, size1], key)
    b_fcN[i] = bias_variable([size1], key)
    h_fcN[i] = tf.nn.tanh(tf.matmul(h_fcN[i-1], W_fcN[i]) + b_fcN[i])

W_fc1 = weight_variable([size1, size2], key)
b_fc1 = bias_variable([size2], key)
freq_estimate = tf.nn.tanh(tf.matmul(h_fcN[nFC-1], W_fc1) + b_fc1)
loss = tf.losses.mean_squared_error(img, decoded_img)



loss = None
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

for i in range(n_training_steps):
    params, freqs = data_batch(n_batch)
    train_step.run(feed_dict={parameters: params, frequencies: freq})
