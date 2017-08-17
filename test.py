import numpy as np
import tensorflow as tf
from bilinear_sampler import bilinear_sampler

x = tf.ones([2, 10, 10, 3])
v = tf.ones([2, 10, 10, 2]) * 2.
y = bilinear_sampler(x, v, resize=True, crop=(5,10,5,10))
z = bilinear_sampler(x, v)

with tf.Session() as sess:
  x_ = sess.run(x)
  y_ = sess.run(y)
  z_ = sess.run(z)

  print x_[0,:,:,0]
  print y_[1,:,:,2]
  print z_[1,:,:,2]
