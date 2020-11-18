#tensorFlow
import tensorflow as tf

sess = tf.InteractiveSession()
indices = tf.constant([[0, 1], [2, 3]])
updates = tf.constant([[5, 5, 5, 5],
                       [6, 6, 6, 6]])
shape = tf.constant([4, 4, 4])
scatter = tf.scatter_nd(indices, updates, shape)
print("TensorFlow")
print(sess.run(scatter), end = "\n\n")

