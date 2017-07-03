# tf-bilinear_sampler
Tensorflow implementation of bilinear sampler with vector field

## Usage
```python
"""
  Args:
    x - Input tensor [N, H, W, C]
    v - Vector flow tensor [N, H, W, 2], tf.float32
"""
y = bilinear_sampler(x, v)

```

## References
- [TensorFlow STN implementation](https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py)
- [Modules for spatial transformer networks (BHWD layout)](https://github.com/qassemoquab/stnbhwd)
