
import tensorflow as tf
from optimizers import *

# Set model weights
W = tf.Variable([[1.,-14.,60.,-70.]],trainable=False,dtype=tf.float32)


#W = tf.Variable([[1.,2.,3.],[2.,-1.,1.],[3.,0.,-1.]],trainable=False,dtype=tf.float32)
b = tf.Variable(0,trainable=False,dtype=tf.float32)
x = tf.Variable(1,dtype=tf.float32 )
Poly = tf.pow(x,2) - 1

# Construct model
#error = (tf.mul(x, W) - b) # Softmax

# Minimize error using cross entropy
loss = Poly

# Gradient Descent
optim = 'ext_grad'
# BFGS
#optim = 'ext_bfgs'
optim = 'ext_newton'

learning_rate = 0.001
ext_grad_learning_rate = 0.01

if(optim == 'ext_grad'):
    optimizer = ExternalPythonGradientDescentOptimizer(loss)
    optimizer.learning_rate = ext_grad_learning_rate
elif(optim == 'ext_bfgs'):
    optimizer = ExternalBFGSOptimizer(loss)
    optimizer.initialized = False
elif(optim == 'ext_newton'):
    optimizer = ExternalNewtonOptimizer(loss)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    #for  i in range(10000):
    #    session.run(optimizer)
    #feed_dict = {input: np.}
    errorVal = session.run(loss)
    print(errorVal)
    errorVal = session.run(W)
    for i in range(1000):
        optimizer.minimize(session)
        errorVal = session.run(loss)
        #finalX = session.run(x)
        #print(finalCost)
        print(errorVal)
