import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 2 + 1

# create tensorflow structure start

Weights = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
biases = tf.Variable(tf.zeros([1]))

y = x_data * Weights + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# create tensorflow structure end

sess = tf.Session()
sess.run(init)          #Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
