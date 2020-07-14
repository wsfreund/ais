import tensorflow as tf
import numpy as np

from .misc import *

math = tf.math

class ObservationModel(object):
  def __call__(self, x, xprime):
    raise NotImplementedError("Overload ObservationModel with one observation model")

  @tf.function
  def comp_avg_logp(self, logp, axis=None):
    """
    Compute the average of log-probabilities using tensorflow algorithms

    """
    # TODO: validatation
    # stabilize numerically by scaling terms
    m = math.reduce_max(logp, axis=axis, keepdims=True)
    logp = math.subtract(logp, m)
    # get probabilities in linear space
    p = math.exp(logp)
    # compute the average
    avgp = math.reduce_mean(p, axis=axis, keepdims=True)
    # return to log space
    avg_logp = math.log(avgp)
    # add common contribution to the average
    return math.add(m, avg_logp)

class GaussianObservationModel(ObservationModel):
  """
  Computes p(x|z) employing the Gaussian assumption, i.e.

  p(x|z) = p_(x;G(z),sigma) = N(x;xprime,sigma)
  """

  def __init__(self, sigma):
    self._sigma = tf.constant( sigma, tf.float32 )
    self._k     = tf.constant( -(np.log(self._sigma) + np.log(2.0*np.pi))/2, tf.float32 )

  @tf.function
  def __call__(self, x, xprime):
    """
    Calculate the logpdf.
    :param x: x states in the posterior p(x|xprime). shape: [num_samples_data, output_dim1, output_dim2]
    :param xprime: xprime states in posterior p(x|xprime). shape: [nchains, num_samples_data, output_dim2, output_dim2]
    """
    const = self._const_gauss_term(xprime) # 1
    logp = self._inner_gauss_term(x,xprime)
    return math.add(logp, const) # nchains, num_samples_data

  @tf.function
  def _inner_gauss_term(self, x, xprime):
    """
    Helper function returning the log-probabilities of x in a gaussian density mixture
    """
    term = self._expand_subtract(x,xprime) # nchains, num_samples_data, output_dim1, output_dim2
    # squared norm2 of diff
    term = math.reduce_sum(math.square(term),axis=(-1,-2,)) # nchains, num_samples_data
    # scale inner term by sigma
    term = math.divide(term,self._sigma) # nchains, num_samples_data
    # scale inner term by 1/2
    return math.multiply(-0.5, term) # nchains, num_samples_data

  @tf.function
  def _expand_subtract(self, x, xprime):
    """
    Helper function returning the log-probabilities of x considering each gaussian component centered at xprime
    """
    expanded_x      = tf.expand_dims(x, 0)              # 1, num_samples_data, output_dim1, output_dim2
    term            = math.subtract(expanded_x,xprime)  # nchains, num_samples_data, output_dim1, output_dim2
    return term

  @tf.function
  def _const_gauss_term(self, xprime):
    """
    Helper function returning constant term of each gaussians
    """
    # compute const term in log-gaussian
    dim = tf.cast( math.reduce_prod( tf.shape(xprime)[-2:]), tf.float32 )
    const = tf.multiply(self._k, dim)
    return const

