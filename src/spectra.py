import numpy as np
import matplotlib.pyplot as plt
from src.modulation import gmsk_mod

def plotWelchPSD(x, fs, fc, ax = None, color = 'b', label = None):
  """
  PLot PSD of a carrier modulated signal using Welch estimate
  Parameters:
    x : signal vector (numpy array) for which the PSD is plotted
    fs : sampling frequency
    fc : center carrier frequency of the signal
    ax : Metplotlib axes object reference for plotting
    color : color character (format string) for the plot
  """
  from scipy.signal import welch
  from scipy.signal.windows import hann
  nx = max(x.shape)
  na = 16 # Averaging factor to plot averaged welch spectrum
  w = hann(nx//na)
  # Welch PSD estimate with Hanning window and no overlap
  f, Pxx = welch(x, fs, window = w, noverlap = 0)
  indices = (f>=fc) & (f<4*fc) # To plot PSD from Fc to 4*Fc
  Pxx = Pxx[indices]/Pxx[indices][0] # Normalized PSD with relation to Fc
  if ax is None:
      plt.plot(f[indices] - fc, 10 * np.log10(Pxx), color, label=label)
  else:
      ax.plot(f[indices] - fc, 10 * np.log10(Pxx), color, label=label)


def gmsk_psd():
  N = 10000 # Number of symbols to transmit
  fc = 146e6 # carrier frequency in Hertz
  L = 16 # oversampling factor,use L= Fs/Fc, where Fs >> 2xFc
  fs = L*fc
  a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's

  #'_':unused output variable
  (s1 , _ ) = gmsk_mod(a,fc,L,BT=0.3, enable_plot=True) # BT_b=0.3
  (s2 , _ ) = gmsk_mod(a,fc,L,BT=0.5) # BT_b=0.5
  (s3 , _ ) = gmsk_mod(a,fc,L,BT=0.7) # BT_b=0.7
  (s4 , _ ) = gmsk_mod(a,fc,L,BT=10000) # BT_b=very value value (MSK)

  # Compute and plot PSDs for each of the modulated versions
  fig, ax = plt.subplots(1, 1)
  plotWelchPSD(s1,fs,fc, ax = ax , color = 'r', label = '$BT_b=0.3$')
  plotWelchPSD(s2,fs,fc, ax = ax , color = 'b', label = '$BT_b=0.5$')
  plotWelchPSD(s3,fs,fc, ax = ax , color = 'm', label = '$BT_b=0.7$')
  plotWelchPSD(s4,fs,fc, ax = ax , color = 'k', label = '$BT_b=\infty$')
  ax.set_xlabel('$f-f_c$'); ax.set_ylabel('PSD (dB/Hz)')
  ax.legend()
  fig.show()

def plot_spectrogram(x, fs, fc, title):
  """
  Plot spectrogram of a carrier modulated signal
  Parameters:
    x : signal vector (numpy array) for which the spectrogram is plotted
    fs : sampling frequency
    fc : center carrier frequency of the signal
  """
  from scipy.signal import spectrogram
  f, t, Sxx = spectrogram(x, fs, nperseg=1024, noverlap=512)
  indices = (f>=0) & (f<4*fc) # To plot PSD from Fc to 4*Fc
  plt.pcolormesh(t, f[indices]-fc, 10 * np.log10(Sxx[indices, :]), shading='gouraud')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title(title)
  plt.colorbar(label='Intensity [dB]')