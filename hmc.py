import tensorflow as tf

from .misc import *

class MCMC(Iterable):
  pass

class HMC(MCMC):

  def __init__(self, energy_fn, **kw ):
    """
    :param energy_f:
    Optional params
    :param stepsize: starting stepsize
    :param n_steps: number of leapfrog steps
    :param target_acceptance_rate:
    :param avg_acceptance_slowness:
    :param stepsize_min:
    :param stepsize_max:
    :param stepsize_dec:
    :param stepsize_inc:
    """
    self._energy_fn = energy_fn
    self._stepsize                = tf.constant( retrieve_kw(kw, 'stepsize',                0.01 ), dtype=tf.float32 )
    self._n_steps                 = tf.constant( retrieve_kw(kw, 'n_steps',                 10   ), dtype=tf.int32   )
    self._stepsize_min            = tf.constant( retrieve_kw(kw, 'stepsize_min',            1e-4 ), dtype=tf.float32 )
    self._stepsize_max            = tf.constant( retrieve_kw(kw, 'stepsize_max',            5e-1 ), dtype=tf.float32 )
    self._stepsize_dec            = tf.constant( retrieve_kw(kw, 'stepsize_dec',            .98  ), dtype=tf.float32 )
    self._stepsize_inc            = tf.constant( retrieve_kw(kw, 'stepsize_inc',            1.02 ), dtype=tf.float32 )
    self._target_acceptance_rate  = tf.constant( retrieve_kw(kw, 'target_acceptance_rate',  0.65 ), dtype=tf.float32 )
    self._avg_acceptance_slowness = tf.constant( retrieve_kw(kw, 'avg_acceptance_slowness', 0.9  ), dtype=tf.float32 )
    # Variables needed at compile time for all tf.functions
    self._v_stepsize              = tf.Variable(0, dtype=tf.float32)
    self._v_avg_acceptance_rate   = tf.Variable(0, dtype=tf.float32)

  @tf.function
  def update(self, initial_pos):
    accept, final_pos, _ = self._move( initial_pos )
    # Compute new position
    new_pos                     = tf.where(accept, final_pos, initial_pos)
    # Update stepsize
    new_stepsize                = tf.multiply( tf.where( tf.greater(self._v_avg_acceptance_rate, self._target_acceptance_rate)
                                             , self._stepsize_inc
                                             , self._stepsize_dec ), self._v_stepsize )
    new_stepsize                = tf.maximum(tf.minimum(new_stepsize, self._stepsize_max), self._stepsize_min)
    self._v_stepsize.assign( new_stepsize )
    # Update acceptance
    new_acceptance_rate         = tf.add( tf.multiply( self._avg_acceptance_slowness, self._v_avg_acceptance_rate )
                                        , tf.multiply( tf.subtract(1.0, self._avg_acceptance_slowness),
                                            tf.reduce_mean(tf.cast(accept,tf.float32))
                                          )
                                        )
    self._v_avg_acceptance_rate.assign( new_acceptance_rate )
    return new_pos

  @tf.function
  def _move(self, initial_pos):
    # NOTE: Initial velocity distribution might have relationship with the
    # latent space distribution. To keep an eye if using a different latent
    # space.
    initial_vel = tf.random.normal(tf.shape(initial_pos))
    final_pos, final_vel = self._simulate_dynamics(initial_pos, initial_vel)
    accept = self._metropolis_hastings_accept(
        energy_prev=self._hamiltonian(initial_pos, initial_vel),
        energy_next=self._hamiltonian(final_pos, final_vel)
    )
    accept = tf.expand_dims(accept,-1)
    return accept, final_pos, final_vel

  @tf.function
  def _hamiltonian(self, p, v):
    return tf.add(self._energy_fn(p), self._kinetic_energy(v))

  @tf.function
  def _kinetic_energy(self, v):
    return tf.multiply(0.5, tf.reduce_sum(tf.multiply(v, v), axis=1) )

  @tf.function
  def _metropolis_hastings_accept(self, energy_prev, energy_next):
    ediff = tf.subtract(energy_prev, energy_next)
    rand = tf.random.uniform(tf.shape(energy_prev))
    accept = tf.math.greater_equal( 
          tf.subtract( tf.exp(ediff), rand )
        , tf.constant(0.0) 
    )
    return accept

  @tf.function
  def _energy_grad(self, pos):
    with tf.GradientTape() as energy_tape:
      energy_tape.watch(pos)
      energy = self._energy_fn(pos)
      total_energy = tf.reduce_sum(energy)
      dE_dpos = energy_tape.gradient(total_energy, pos)
    return dE_dpos

  @tf.function
  def _dynamics_leapfrog(self, pos, vel, i):
    dE_dpos = self._energy_grad(pos)
    new_vel = tf.subtract(vel, tf.multiply( self._v_stepsize, dE_dpos ) )
    new_pos = tf.add(pos, tf.multiply(self._v_stepsize, new_vel ) )
    return [new_pos, new_vel, tf.add(i, 1)]

  def _dynamics_condition(self, pos, vel, i):
    return tf.less(i, self._n_steps)

  @tf.function
  def _simulate_dynamics(self, initial_pos, initial_vel):
    dE_dpos = self._energy_grad(initial_pos)
    vel_half_step = tf.subtract(initial_vel, tf.multiply( tf.multiply(0.5, self._v_stepsize), dE_dpos ) )
    pos_full_step = tf.add(initial_pos, tf.multiply( self._v_stepsize, vel_half_step ) )
    i = tf.constant(0)
    final_pos, new_vel, _ = tf.while_loop( self._dynamics_condition
                                         , self._dynamics_leapfrog
                                         , [pos_full_step, vel_half_step, i])
    dE_dpos = self._energy_grad(final_pos)
    final_vel = tf.subtract( new_vel, tf.multiply(0.5, tf.multiply( self._v_stepsize, dE_dpos ) ) )
    return final_pos, final_vel

  def __enter__(self):
    # reset or initialize iterative properties
    self._v_stepsize.assign( self._stepsize )
    self._v_avg_acceptance_rate.assign( self._target_acceptance_rate )
