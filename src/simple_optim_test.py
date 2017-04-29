
import tensorflow as tf
from optimizers import *

# Set model weights
W = tf.Variable([[1.,2.,3.],[2.,-1.,0.],[3.,1.,-1.]],trainable=False,dtype=tf.float32, name='jacobian')
#W = tf.Variable([[1.,2.,3.],[2.,-1.,1.],[3.,0.,-1.]],trainable=False,dtype=tf.float32)
b = tf.Variable([9.,8.,3.],trainable=False,dtype=tf.float32)
x = tf.Variable([[1.,1.,1.]],dtype=tf.float32 )
# Construct model
error = (tf.matmul(x, W) - b)

# Minimize error using cross entropy
loss = tf.reduce_mean(tf.reduce_sum(tf.square(error)))

# Gradient Descent
#optim = 'ext_grad'
# BFGS
#optim = 'ext_bfgs'
#optim = 'ext_newton'

#optim = 'ext_d_simplex'

learning_rate = 0.001
ext_grad_learning_rate = 0.01
adam_learning_rate = 0.001
optim = 'adam'


if(optim == 'ext_grad'):
    optimizer = ExternalPythonGradientDescentOptimizer(loss)
    optimizer.learning_rate = ext_grad_learning_rate
elif(optim == 'ext_bfgs'):
    optimizer = ExternalBFGSOptimizer(loss)
    optimizer.initialized = False
elif(optim == 'ext_newton'):
    # for this problem, the weight matrix W *is* the jacobian at all values
    optimizer = ExternalNewtonOptimizer(error, W)
elif(optim == 'ext_d_simplex'):
    # for this problem, the weight matrix W *is* the jacobian at all values
    optimizer = ExternalDownhillSimplexOptimizer(loss)
elif(optim == 'adam'):
    optimizer = tf.train.AdamOptimizer(learning_rate= adam_learning_rate).minimize(loss)

init = tf.global_variables_initializer()
trainingIter = 10000
i = 0
with tf.Session() as session:
    session.run(init)

    if (optim == 'adam'):
        while(i < trainingIter):
            session.run(optimizer)
            i = i+1
            if (trainingIter % 1000 == 0):
                finalX = session.run(x)
                print(finalX)

    else:
        #for  i in range(10000):
        #    session.run(optimizer)
        #feed_dict = {input: np.}
        errorVal = session.run(error)
        print(errorVal)
        errorVal = session.run(W)
        for i in range(1000):
            optimizer.minimize(session)
            #errorVal = session.run(error)
            finalX = session.run(x)
            print(finalX)
            #print(errorVal)
