import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .misc import *
from .hmc import *
from .observation import *

class Scheduler(object):
  def __init__(self, n_points):
    self.n_iterations = n_points - 1

  def __len__(self):
    return self.n_iterations if self.n_iterations > 1 else 1

class SigmoidalScheduler(Scheduler):

  def __init__(self, n_points, rad=4):
    Scheduler.__init__(self,n_points)
    self.rad = rad

  def __call__(self):
    if self.n_iterations == 1:
      return np.array([0.0, 1.0])
    t = np.linspace(-self.rad, self.rad, self.n_iterations+1)
    s = 1.0 / (1.0 + np.exp(-t))
    return (s - np.min(s)) / (np.max(s) - np.min(s))

def np_avg_logexp(A,axis=None):
  A_max = np.max(A, axis=axis, keepdims=True)
  B = (
      np.log(np.mean(np.exp(A - A_max), axis=axis, keepdims=True)) +
      A_max
  )
  return B 


class AIS(Iterable):

  def __init__( self, generator
              , mcmc_type = HMC
              , observation_model_type = GaussianObservationModel
              , scheduler_type = SigmoidalScheduler
              , **kw ):
    """
    Implements Annealed Importance Sampling.

    :param generator: 
    It must provide the following methods:
      - `z = generator.sample_latent_data(nsamp)' 
      - `xprime = generator.transform(z)'
      - `prior_logp = generator.log_prob(z)'
    :param mcmc: A Markov Chain Monte Carlo type
    :param observation_model: Observation model providing p(x|xprime)
    :param schedule: Updates of the AIS geometric average beta parameter 
    """
    assert issubclass(mcmc_type, MCMC), "Not allowed MCMC type"
    assert issubclass(observation_model_type, ObservationModel), "Not allowed observation model type"
    assert issubclass(scheduler_type, Scheduler), "Not allowed schedule type"
    self._generator          = generator
    self._mcmc               = mcmc_type( self._energy_fn, **retrieve_kw(kw, 'mcmc_args',              {} ) )
    self._observation_model  = observation_model_type(     **retrieve_kw(kw, 'observation_model_args', {} ) )
    self._scheduler          = scheduler_type(             **retrieve_kw(kw, 'schedule_args',          {} ) )
    self._nchains            = retrieve_kw( kw, 'nchains', 16 )

  def forward_ais(self, x):
    """
    :param x: positions in the data input space
    :return: an array with a lower bound on logp for each x and its average
    """
    with self:
      # prepare to loop
      schedule = self._scheduler()
      # initialize looping variables
      self._l_const_ndata = x.shape[0]
      self._l_const_data_dim = x.shape[1:]
      self._l_const_x = tf.constant(x)
      # Expand schedule with the values to be used in each iteration
      self._l_const_items = tf.unstack(
          tf.convert_to_tensor([[t0, t1] for (t0, t1) in 
            zip(schedule[:-1], schedule[1:])
          ], dtype=tf.float32)
      )
      self._l_t0 = tf.Variable( 0., dtype = tf.float32 )
      self._l_t1 = tf.Variable( 0., dtype = tf.float32 )
      # We will work on flattened space until we get the final results.  Only
      # exception is the observation model, which will use the matrix
      # representation in order to broadcast operations correctly in the input
      # space
      self._l_z = tf.Variable(
        self._generator.sample_latent_data(
          self._nchains*self._l_const_ndata  
        ) 
      )
      self._l_w = tf.Variable( tf.zeros( self._nchains*self._l_const_ndata, dtype = tf.float32 ) )
      with self._mcmc:
        # loop
        tf.while_loop( self._condition
                     , self._body
                     , (tf.constant(0),) # iteration counter and thread-safe MCMC
                     , parallel_iterations=1
                     , swap_memory=True )
        # Unflatten final results
        desired_shape = ( self._nchains, self._l_const_ndata )
        final_state = tf.reshape( self._l_z, desired_shape + (self._generator.latent_dim(),) )
        lower_bound_logp = tf.reshape( self._l_w, desired_shape )
      #
    #
    # Compute statistics
    avg_lower_bound_logp = np_avg_logexp(lower_bound_logp, axis = 0)
    final_lld = np.mean(avg_lower_bound_logp)
    return final_lld, final_state, lower_bound_logp, avg_lower_bound_logp

  @tf.function
  def _log_f_i(self, t):
    return tf.multiply( -1.0, self._energy_fn( self._l_z, t) )

  # TODO This method can be improved when used with VAE models
  @tf.function
  def _energy_fn(self, z, t = None):
    if t is None: t = self._l_t1
    prior = self._generator.latent_log_prob(z)
    posterior = tf.multiply( t,  
      # Flatten observation results
      tf.reshape( 
        self._observation_model(
          self._l_const_x, 
          # Unflatten generated data so that observation model
          # can understand broadcast operations on the input space
          tf.reshape( self._generator.transform(z)
            , ( self._nchains, self._l_const_ndata, ) + self._l_const_data_dim
          )
        ), ( self._nchains * self._l_const_ndata, )
      )
    )
    energy = tf.multiply( -1.0, tf.add( prior, posterior ) )
    return energy

  @tf.function
  def _update_w(self): 
    new_u = self._log_f_i(self._l_t1)
    prev_u = self._log_f_i(self._l_t0)
    new_logp = self._l_w.assign_add( tf.subtract(new_u, prev_u) )
    return new_logp

  @tf.function
  def _body(self, *args):
    # Retrieve new interation information
    index = args[0]
    item = tf.gather(self._l_const_items, index)
    self._l_t0.assign( tf.gather(item, 0) )
    self._l_t1.assign( tf.gather(item, 1) )
    # Update w
    self._update_w()
    # Update z
    self._l_z.assign( self._mcmc.update( self._l_z ) )
    # Update looping status
    index = tf.add(index,1)
    return (index,)

  def _condition(self, *args):
    index = args[0]
    return tf.less(index, self._l_const_len_schedule)

  def __enter__(self):
    self._l_const_len_schedule = len(self._scheduler)
