import tensorflow as tf
import numpy as np

def bilinear_sampler(x, v, resize=False, normalize=False, crop=None):
  """
    Args:
      x - Input tensor [N, H, W, C]
      v - Vector flow tensor [N, H, W, 2], tf.float32

      (optional)
      resize - Whether to resize v as same size as x
      normalize - Whether to normalize v from scale 1 to H (or W).
                  h : [-1, 1] -> [-H/2, H/2]
                  w : [-1, 1] -> [-W/2, W/2]
      crop - Set the region to sample. 4-d list [h0, h1, w0, w1]
  """

  def _get_grid_array(N, H, W, h, w):
    N_i = np.arange(N)
    H_i = np.arange(h, h+H)
    W_i = np.arange(w, w+W)
    n, h, w, = np.meshgrid(N_i, H_i, W_i, indexing='ij')
    n = np.expand_dims(n, axis=3)
    h = np.expand_dims(h, axis=3)
    w = np.expand_dims(w, axis=3)
    return np.concatenate([n, h, w], axis=3)

  shape = x.get_shape().as_list() # Should it be fixed size ?
  N = shape[0]
  if crop is None: 
    H = shape[1]
    W = shape[2]
    h = w = 0
  else :
    H = crop[1] - crop[0]
    W = crop[3] - crop[2]
    h = crop[0]
    w = crop[2]

  if resize:
    if callable(resize) :
      v = resize(v, [H, W])
    else :
      v = tf.image.resize_bilinear(v, [H, W])

  vy, vx = tf.split(v, 2, axis=3)
  if normalize :
    vy *= (H / 2)
    vx *= (W / 2)

  vx0 = tf.floor(vx)
  vy0 = tf.floor(vy)
  vx1 = vx0 + 1
  vy1 = vy0 + 1 # [N, H, W, 1]

  v00 = tf.concat([vy0, vx0], 3)
  v01 = tf.concat([vy0, vx1], 3)
  v10 = tf.concat([vy1, vx0], 3)
  v11 = tf.concat([vy1, vx1], 3) # [N, H, W, 2]

  padding = [[0, 0], [0, 0], [0, 0], [1, 0]]
  v00 = tf.pad(v00, padding, mode='CONSTANT')
  v01 = tf.pad(v01, padding, mode='CONSTANT')
  v10 = tf.pad(v10, padding, mode='CONSTANT')
  v11 = tf.pad(v11, padding, mode='CONSTANT') # [N, H, W, 3]

  idx = _get_grid_array(N, H, W, h, w) # [N, H, W, 3]
  idx00 = tf.cast(v00 + idx, tf.int32)
  idx01 = tf.cast(v01 + idx, tf.int32)
  idx10 = tf.cast(v10 + idx, tf.int32)
  idx11 = tf.cast(v11 + idx, tf.int32)

  x00 = tf.gather_nd(x, idx00)
  x01 = tf.gather_nd(x, idx01)
  x10 = tf.gather_nd(x, idx10)
  x11 = tf.gather_nd(x, idx11)

  w00 = tf.cast((vx1 - vx) * (vy1 - vy), tf.float32)
  w01 = tf.cast((vx1 - vx) * (vy - vy0), tf.float32)
  w10 = tf.cast((vx - vx0) * (vy1 - vy), tf.float32)
  w11 = tf.cast((vx - vx0) * (vy - vy0), tf.float32)
  output = tf.add_n([w00*x00, w01*x01, w10*x10, w11*x11])

  return output
