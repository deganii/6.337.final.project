
import tensorflow as tf
from optimizers import *

vector = tf.Variable([7., 7.], 'vector')

# Make vector norm as small as possible.
loss = tf.reduce_sum(tf.square(vector))
learning_rate = 0.001
#optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#optimizer = PurePythonGradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = ExternalPythonGradientDescentOptimizer(loss, options={'maxiter': 100})
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    #for  i in range(10000):
    #    session.run(optimizer)
    #feed_dict = {input: np.}
    for i in range(10000):
        optimizer.minimize(session)
    finalVal = session.run(vector)
    print(finalVal)
