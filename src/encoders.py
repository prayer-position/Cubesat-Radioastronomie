import numpy as np
from numba import njit

def differential_encode(bits):
  """
  Encodes bits based on transitions
  y[n] = x[n] XOR y[n-1]
  """
  out = np.zeros_like(bits)
  out[0] = bits[0]
  for i in range(1, len(bits)):
    out[i] = bits[i] ^ out[i-1]
  return out

def differential_decode(bits):
  """
  Decodes transitions back to original bits
  x[n] = y[n] XOR y[n-1]
  """
  out = np.zeros_like(bits)
  out[0] = bits[0]
  for i in range(1, len(bits)):
    out[i] = bits[i] ^ bits[i-1]
  return out

# @njit tells Python to compile this directly into C code
@njit
def viterbi(received_iq):
    """
    Decodes a baseband I/Q signal using a 4-state Viterbi algorithm
    Expects 1 sample per symbol (downsampled from L oversampling)
    """
    ideal_real = np.array([1.0, 0.0, -1.0, 0.0])
    ideal_imag = np.array([0.0, 1.0, 0.0, -1.0])
    num_states = 4

    # Trellis map: From each state, where can we go ?
    # Format: current_state: {bit_value: next_stage}
    # Bit 1 = +90 deg (state + 1), Bit 0 = -90 deg (state -1)
    transitions = np.array([
       [3, 1],
       [0, 2],
       [1, 3],
       [2, 0]
    ], dtype = np.int32)

    # Path Metrics initialization (cost to reach a state)
    # Start at state 0 with 0 cost & others at high value
    path_metrics = np.array([0.0, 1e9, 1e9, 1e9])

    # Keep track of the surviving paths: {state: [list of bits]}
    traceback_states = np.zeros((len(received_iq), 4), dtype = np.int32)
    traceback_bits = np.zeros((len(received_iq), 4), dtype = np.int8)

    # Looping through every received I/Q symmbol
    for i in range(len(received_iq)):
        r_real = received_iq[i].real
        r_imag = received_iq[i].imag

        new_metrics = np.array([1e9, 1e9, 1e9, 1e9])

        for state in range(4):
           if path_metrics[state] > 1e8:
              continue # Skip dead paths
           
           for bit in range(2):
              next_state = transitions[state, bit]

              # Fast Euclidean distance calculation
              dist = (r_real - ideal_real[next_state])**2 + (r_imag - ideal_imag[next_state])**2
              cost = path_metrics[state] + dist

              if cost < new_metrics[next_state]:
                 new_metrics[next_state] = cost
                 traceback_states[i, next_state] = state
                 traceback_bits[i, next_state] = bit

        path_metrics = new_metrics
    
    out_bits = np.zeros(len(received_iq), dtype=np.int8)
    
    # Find the best ending state
    best_state = 0
    min_metric = path_metrics[0]
    for s in range(1, 4):
       if path_metrics[s] < min_metric:
          min_metric = path_metrics[s]
          best_state = s
      
    # Trace the bath backwards to recover the bits
    curr_state = best_state
    for i in range(len(received_iq) -1, -1, -1):
       out_bits[i] = traceback_bits[i, curr_state]
       curr_state = traceback_states[i, curr_state]
    
    return out_bits