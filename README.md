# tf-bilinear_sampler
Tensorflow implementation of bilinear sampler with vector field

## Usage
```python
"""
  Args:
    x - Input tensor [N, H, W, C]
    v - Vector flow tensor [N, H, W, 2], tf.float32

    (optional)
    resize - Whether to resize v as same size as x
    normalize - Whether to normalize v from scale 1 to H/2 (or W/2).
                h : [-1, 1] -> [-H/2, H/2]
                w : [-1, 1] -> [-W/2, W/2]
    crop - Set the region to sample. 4-d list [h0, h1, w0, w1]
    out  - Handling out of boundary value.
           Zero value is used if out="CONSTANT". (default)
           Boundary values are used if out="EDGE".
"""
y = bilinear_sampler(x, v)

```

## References
- [TensorFlow STN implementation](https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py)
- [Modules for spatial transformer networks (BHWD layout)](https://github.com/qassemoquab/stnbhwd)
