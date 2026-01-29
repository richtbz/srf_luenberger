# Copyright 2024 A. Bellandi et al.
# Copyright 2025 B. Richter et al.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
  Luenberger observer simulation routine.
  See: B. Richter et al. 'Estimation of superconducting cavity
  bandwidth and detuning using a Luenberger observer.'
"""

import numpy as np

def luenberger_observer(
    probe, forward, half_bandwidth_ext, detuning_init,
    sample_rate, observer_bandwidth, amplitude_threshold,
    bandwidth_gain_factor = 1, detuning_gain_factor = 1,
    I_init = 0.0, Q_init = 0.0, show_factor_limits=False):
  """Simulate the Luenberger observer.
  Internally, bandwidth and detuning estimation is done with
  normalization by the external half bandwidth.

  Args:
    probe (np.ndarray):
      2D array with two columns that represents the cavity probe
      signal trace in I/Q. The first column is the in-phase part
      and the second one is the quadrature.

    forward (np.ndarray):
      2D array with two columns that represents the cavity forward
      signal trace in I/Q. The first column is the in-phase part
      and the second one is the quadrature.

    half_bandwidth_ext (float):
      External half bandwidth of the cavity in Hz.

    detuning_init (float):
      Initial detuning in Hz.

    sample_rate (float):
      System sample rate in Hz.

    observer_bandwidth (float):
      Observer bandwidth in Hz.

    amplitude_threshold (float):
      Minimum absolute probe amplitude for which the adaptation of
      bandwidth and detuning is active.

    bandwidth_gain_factor (float, optional)
      Excess bandwidth estimation bandwidth adjustment.
      Defaults to 1.

    detuning_gain_factor (float, optional)
      Detuning estimation bandwidth adjustment.
      Defaults to 1.

    I_init (float, optional):
      Initial inphase RF observer state in MV.
      Defaults to 0.

    Q_init (float, optional):
      Initial quadrature RF observer state in MV.
      Defaults to 0.

  Returns:
    (np.ndarray):
      2D array with four columns of the estimated values.
      The first two columns represents the estimated I/Q signals
      for the probe.
      The third column represents the excess bandwidth. The total
      bandwidth results from adding the external half bandwidth
      with this column.
      The fourth column represents the detuning.
  """

  samples = forward.shape[0]
  T = 1. / sample_rate

  amplitude_square_minimum = amplitude_threshold**2

  ## Initialize discrete time LTV model. Eq. (12)
  sys_diagonal = np.exp(-2.0 * np.pi * half_bandwidth_ext * T)
  alpha = 1.0 - sys_diagonal

  Phi = np.zeros((4, 4))
  Phi[0, 0] = Phi[1, 1] = sys_diagonal
  Phi[2, 2] = Phi[3, 3] = 1.

  Gam = np.zeros((4, 2))
  Gam[0, 0] = Gam[1, 1] = 2. * alpha

  C = np.zeros((2, 4))
  C[0, 0] = C[1, 1] = 1.


  ## Initialize gain matrix. Eqs. (C6, C12)
  rho = np.exp(-2. * np.pi * observer_bandwidth * T)
  mu_0 = -(1-rho)**2 / alpha
  mu_1 = bandwidth_gain_factor * mu_0
  mu_2 = detuning_gain_factor * mu_0

  Lambda = np.zeros((4, 2))
  Lambda[0, 0] = Lambda[1, 1] = 2. * rho - 1. - sys_diagonal

  # Display tuning factor limits. Eq. (C11)
  kappa = 10
  if show_factor_limits:
      print(f"butterworth phi approx {0.33*rho+1.11:.2f}")
      print(1 - (
        (np.exp(-2*np.pi*kappa*half_bandwidth_ext*T)-rho)/
        (1 - rho)
        )**2, " < phi <", 2/(1-rho))

  ## Set initial estimated state vector.
  x_estimate = np.zeros((samples, 4))
  x_estimate[0, 0] = I_init
  x_estimate[0, 1] = Q_init
  x_estimate[0, 3] = detuning_init / half_bandwidth_ext


  ## Reserve memory for predicted state vector
  x_predict = np.zeros((4))

  for i in range(1, samples):

    # Prediction step

    ## Update the system matrix coefficients. Eq. (12)
    Phi[0, 2] = -alpha * x_estimate[i-1, 0]
    Phi[0, 3] = -alpha * x_estimate[i-1, 1]
    Phi[1, 2] = Phi[0, 3]
    Phi[1, 3] = -Phi[0, 2]

    ## State prediction
    x_predict = (np.dot(Phi, x_estimate[i-1, :]) +
                 np.dot(Gam, forward[i-1, :]))

    # Update step

    ## State estimation Eqs. (5, 6)
    prediction_error = probe[i, :] - np.dot(C, x_predict)
    x_estimate[i, :] = (x_predict -
      np.dot(Lambda, prediction_error))

    ## Update the observer gain coefficients. (C12)

    ampl_sq = x_estimate[i, 0]**2 + x_estimate[i, 1]**2

    ## Check square amplitude condition
    if ampl_sq > amplitude_square_minimum:
      bandwidth_coefficient = mu_1 / ampl_sq
      detuning_coefficient = mu_2 / ampl_sq
    else:
      # Disable detuning and bandwidth update
      bandwidth_coefficient = 0.0
      detuning_coefficient = 0.0

    ## Update the gain matrix. Eq. (12)
    Lambda[2, 0] = -bandwidth_coefficient * x_estimate[i, 0]
    Lambda[2, 1] = -bandwidth_coefficient * x_estimate[i, 1]
    Lambda[3, 0] = -detuning_coefficient * x_estimate[i, 1]
    Lambda[3, 1] = detuning_coefficient * x_estimate[i, 0]

  # Reverse the normalization.
  x_estimate[:, 2] *= half_bandwidth_ext
  x_estimate[:, 3] *= -half_bandwidth_ext
  return x_estimate
