import numpy as np  # Move this to the top!
import matplotlib.pyplot as plt
from scipy.signal import upfirdn, lfilter

def gaussianLPF(BT, Tb, L, k):
  """
  Generate filter coefficients of Gaussian low pass filter
  Parameters:
    BT : BT prouct - Bandwidth x bit period
    Tb : bit period
    L  : oversampling factor (number of samples per bit)
    k  : span length of the pulse (bit interval)
  Returns:
    h_norm : normalized filter coefficients of Gaussian LPF
  """

  B = BT/Tb
  # Truncated time limits for the filter
  t = np.arange(start = -k*Tb, stop = k*Tb + Tb/L, step = Tb/L)
  h = B*np.sqrt(2*np.pi/(np.log(2)))*np.exp(-2 * (t*np.pi*B)**2 /(np.log(2)))
  h_norm = h/np.sum(h)
  return h_norm


def gmsk_mod(a, fc, L, BT, enable_plot = False):
  """
  Function to modulate a binary stream using GMSK modulation
  Parameters:
      BT : Bandwidth-Time product
      a : input binary data stream to modulation
      fc : RF carrier frequency (Hz)
      L : oversampling factor
      enable_plot : True = plot transmitter waveforms (default False)
  Returns :
      (s_t, s_complex) : tuple containing the following variables
        s_t : GMSK modlated signal with carrier s(t)
        s_complex : baseband GMSK signal
  """

  # Derived waveform timing paramters
  fs = L * fc
  Ts = 1/fs
  Tb = L*Ts
  # NRZ pulse train
  c_t = upfirdn(h = [1]*L, x = 2*a-1, up = L)

  k = 4 #trunc length for Gaussian LPF
  h_t = gaussianLPF(BT, Tb, L, k) # Gaussian LPF
  b_t = np.convolve(h_t, c_t, 'same') # convolve c(t) with Gaussian LPF to get b(t)
  bnorm_t = b_t/max(abs(b_t)) # normalize the output of Gaussian LPF to +/-1

  h = 0.5
  # integrate to get phase information
  phi_t = lfilter(b = [1], a = [1, -1], x = bnorm_t*Ts) * h*np.pi/Tb

  I = np.cos(phi_t)
  Q = np.sin(phi_t)
  s_complex = I - 1j*Q # Complex baseband representation
  t = Ts * np.arange(start = 0, stop = len(I)) # time for RF carrier
  sI_t = I*np.cos(2*np.pi*fc*t)
  sQ_t = Q*np.sin(2*np.pi*fc*t)
  s_t = sI_t - 1j*sQ_t # GMSK modulated signal

  if enable_plot:
    fig, axs = plt.subplots(2, 4)
    axs[0,0].plot(np.arange(0,len(c_t))*Ts,c_t);
    axs[0,0].set_title('c(t)');axs[0,0].set_xlim(0,40*Tb)
    axs[0,1].plot(np.arange(-k*Tb,k*Tb+Ts,Ts),h_t);
    axs[0,1].set_title('$h(t): BT_b$='+str(BT))
    axs[0,2].plot(t,I,'--');axs[0,2].plot(t,sI_t,'r');
    axs[0,2].set_title('$I(t)cos(2 \pi f_c t)$');
    axs[0,2].set_xlim(0,10*Tb)
    axs[0,3].plot(t,Q,'--');axs[0,3].plot(t,sQ_t,'r');
    axs[0,3].set_title('$Q(t)sin(2 \pi f_c t)$');
    axs[0,3].set_xlim(0,10*Tb)
    axs[1,0].plot( np.arange(0,len(bnorm_t))*Ts,bnorm_t);
    axs[1,0].set_title('b(t)');axs[1,0].set_xlim(0,40*Tb)
    axs[1,1].plot(np.arange(0,len(phi_t))*Ts, phi_t);
    axs[1,1].set_title('$\phi(t)$')
    axs[1,2].plot(t,s_t);
    axs[1,2].set_title('s(t)'); axs[1,2].set_xlim(0,20*Tb)
    axs[1,3].plot(I,Q);axs[1,3].set_title('constellation')
    fig.show()
  return (s_t, s_complex)
