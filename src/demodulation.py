def gmsk_demod(r_complex, L):
  """
  Function to demodulate a baseband GMSK signal
  Parameters :
    r_complex : received signal at receiver front end (complex form - I+Q)
    L : oversampling factor
  Returns :
    a_hat : detected binary stream
  """
  I=np.real(r_complex); Q = -np.imag(r_complex); # I,Q streams
  z1 = Q * np.hstack((np.zeros(L), I[0:len(I)-L]))
  z2 = I * np.hstack((np.zeros(L), Q[0:len(I)-L]))
  z = z1 - z2
  a_hat = (z[L-1::L] > 0).astype(int) #sampling and hard decision
  # Sampling indices depend on the truncation length (k) of Gaussian LPF defined
  # in the modulator
  return a_hat
