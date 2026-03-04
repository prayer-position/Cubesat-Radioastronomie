import numpy as np

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

def viterbi(received_iq):
    """
    Decodes a baseband I/Q signal using a 4-state Viterbi algorithm
    Expects 1 sample per symbol (downsampled from L oversampling)
    """
    ideal_phases = [1+0j, 0+1j, -1+0j, 0-1j]
    num_states = 4

    # Trellis map: From each state, where can we go ?
    # Format: current_state: {bit_value: next_stage}
    # Bit 1 = +90 deg (state + 1), Bit 0 = -90 deg (state -1)
    transitions = {
      0: {1: 1, 0: 3},
      1: {1: 2, 0: 0},
      2: {1: 3, 0: 1},
      3: {1: 0, 0: 2}
    }

    # Path Metrics initialization (cost to reach a state)
    # Start at state 0 with 0 cost & other at infinity
    path_metrics = {0: 0.0, 1: float('inf'), 2: float('inf'), 3: float('inf')}

    # Keep track of the surviving paths: {state: [list of bits]}
    paths = {0: [], 1: [], 2: [], 3: []}

    # Looping through every received I/Q symmbol
    for r in received_iq:
        new_metrics = {0: float('inf'), 1: float('inf'), 2: float('inf'), 3: float('inf')}
        new_paths = {0: [], 1: [], 2: [], 3: []}

        for current_state in range(num_states):
            if path_metrics[current_state] == float('inf'):
                continue # Skip uncreachable states
        
            for bit, next_state in transitions[current_state].items():
                # Expected symbol is the ideal pahse of the next state
                expected_symbol = ideal_phases[next_state]

                # BRANCH METRIC: Squared Euclidean distance between received and expected
                # Soft decision algorithm 
                distance = (np.real(r) - np.real(expected_symbol))**2 + \
                           (np.imag(r) - np.imag(expected_symbol))**2

                # PATH MATRIC: Accumulated cost
                total_cost = path_metrics[current_state] + distance 

                # If new path to 'next_state' is cheaper than the existing one, we keep it
                if total_cost < new_metrics[next_state]:
                    new_metrics[next_state] = total_cost
                    new_paths[next_state] = paths[current_state] + [bit]
        
        path_metrics = new_metrics
        paths = new_paths
    
    # TRACEBACK: Find the state with the lwoest total error at the very end
    best_final_state = min(path_metrics, key = path_metrics.get)
    best_bit_sequence = paths[best_final_state]

    return np.array(best_bit_sequence)