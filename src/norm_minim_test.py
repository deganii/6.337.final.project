
import tensorflow as tf
from optimizers import *

x = tf.Variable([7., 7.], 'vector')


x_var = tf.Variable(x, trainable=False, collections=[],name='x_var',dtype=tf.float32)
y_var = tf.Variable(y, trainable=False, collections=[],name='y_var',dtype=tf.float32)


# Set model weights
W = tf.Variable([[1.,2.,3.,],[2.,-1.,1.],[3.,0.,-1.]],trainable=False,dtype=tf.float32)
b = tf.Variable([9.,8.,3.],trainable=False,dtype=tf.float32)

# Construct model
pred = tf.nn.softmax(tf.matmul(x_var, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y_var*tf.log(pred), reduction_indices=1))
# Gradient Descent


#optim = 'ext_grad'
optim = 'ext_bfgs'

# Make vector norm as small as possible.
loss = 5 + tf.reduce_sum(tf.atan(tf.square(vector)))
learning_rate = 0.001
ext_grad_learning_rate = 0.01

if(optim == 'ext_grad'):
    optimizer = ExternalPythonGradientDescentOptimizer(loss)
    optimizer.learning_rate = ext_grad_learning_rate
elif(optim == 'ext_bfgs'):
    optimizer = ExternalBFGSOptimizer(loss)
    optimizer.initialized = False

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    #for  i in range(10000):
    #    session.run(optimizer)
    #feed_dict = {input: np.}
    for i in range(10):
        optimizer.minimize(session)
    finalVal = session.run(vector)
    print(finalVal)
