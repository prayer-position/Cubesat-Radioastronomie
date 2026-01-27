def awgn(s, SNRdB, L = 1):
  """
  AWGN channel

  add awgn noise to input signal. The function adds AWGN noise vector to signal 
  's' to generate a resulting signal vector 'r' of specifies SNR in dB. It also 
  return the noise vector 'n' that is added to the signal 's' and th power 
  spectral density N0 of noise added

  Parameters : 
    s : input/transmitted signal vector
    SNRdB : desired signal ot nosie ratio (expressed in dB) for the 
        received signal
    L : Oversampling factor (applicable for waveform simulation)
        default L = 1.
  Returns : 
    r : received signal vector (r = s+n)
  """
  from numpy import sqrt, sum, abs, isrealobj
  from numpy.random import standard_normal
  gamma = 10**(SNRdB/10) #SNR to linear scale
  if s.ndim==1:# if s is single dimensional vector
    P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
  else: # multi-dimensional signals like MFSK
    P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
  N0=P/gamma # Find the noise spectral density
  
  if isrealobj(s):# check if input is real/complex object type
    n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
  else:
    n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
  r = s + n # received signal
  return r
