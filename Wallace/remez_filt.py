import soundfile as sf
from scipy import signal 
import numpy as np

def remez_lp_filt(x, fc, fs):
    cutoff = fc    # Desired cutoff frequency, Hz
    trans_width = 200  # Width of transition from pass to stop, Hz
    numtaps = 1000   # Size of the FIR filter.
    taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs],
                        [1, 0], fs=fs)

    x = signal.lfilter(taps, 1, x, axis=-1, zi=None)
    return x

def main():
    x, fs = sf.read('/home/wallace.abreu/Mestrado/behm-gan_vanilla/audio_examples/YK_Track20.wav')
    if x.shape[1] > 1:
        x = np.mean(x, axis=1)
    fc = 3000
    x_filt = remez_lp_filt(x, fc, fs)
    sf.write('/home/wallace.abreu/Mestrado/behm-gan_vanilla/audio_examples/YK_Track20_3k_remez.wav', x_filt, fs)
    
if __name__ == '__main__':
    main()