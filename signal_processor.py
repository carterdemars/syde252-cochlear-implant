import scipy
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd

class SignalProcessor():
    def __init__(self, audio):
        self.audio = audio
        self.audio_data = None # Data is read from audio file and stored here
        self.sample_rate = None #Sampling rate of the audio file

    def get_sampling_rate(self, audio):
        """
        3.1: reads the file first, then finds sampling rate
        The function determines sampling rate in kHz of an input signal.
        :return: sampling rate
        """
        sample_rate, audio = wavfile.read(audio)
        print(f"Sampling Rate of Original WAV file: {sample_rate} Hz")
        return sample_rate, audio
    
    def mono_stereo(self):
        """
        checks whether input sound is stereo or mono 
        if stereo, add both columns to a signal channel (1 column array)
        """
        if len(self.audio_data.shape) == 2:
            self.audio_data = self.audio_data.sum(axis=1)/2 # converts to mono audio from stereo
        print("Coverted from Stereo to Mono")

    def play_sound(self):
        """
        Plays the sound out loud.
        :return: None
        """
        sd.play(self.audio_data, self.sample_rate) #plays data extracted from audio file at found sampling rate
        sd.wait() #blocks python interpreter until playback is finished
        return

    def plot_waveform(self):
        """
        Plots the sound waveform as a function of the sample number
        :return:
        """
        plt.figure(figsize=(10,4))
        plt.plot(self.audio_data)
        plt.title('Audio Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.show()
        return
    
    def resample_audio(self):
        """
        if sample rate of audio is not 16000Hz, funtion resamples it to 16000Hz
        """
        if self.sample_rate != 16000:
            self.audio_data = signal.resample(self.audio_data, int(16000/self.sample_rate * len(self.audio_data)))
            self.sample_rate = 16000
            print("Audio Resampled")
    
    def generate_cos(self):
        """
        Generates a cosine signal of 1kHz frequency that has the same duration as the audio signal
        """
        #duration of signal (total time in s that original audio lasts) = total number of samples/sample rate
        duration = len(self.audio_data) / self.sample_rate

        #creating linearly spaced array for 1kHz
        time = np.linspace(0., duration, len(self.audio_data))

        frequency = 1000 #1kHz

        #generating cosine wave: A * cos(2*pi*frequency*t)
        #where A = amplitude, t = time
        cosine_signal = np.cos(2* np.pi * frequency * time)

        return time, cosine_signal
    

    def plot_cos(self, time, cosine_signal):
        """
        Plots the first two cycles of the generated cos signal
        """
        freq = 1000
        plt.figure(figsize=(10,4))
        #plotting for two cycles:
        #x axis: divide sample rate by frequency, and multiply by 2 to plot 2 cycles
        #y axis: same as above, but with cosine signal
        two_cycles = int((2/freq) * self.sample_rate)
        plt.plot(time[:two_cycles], cosine_signal[:two_cycles])
        plt.title("Cosine Waveform (1kHz)")
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.show()

    def normalize_audio(self):
        """
        Normalize audio to fit withing range of [-1, 1]
        """
        max_audio = np.max(np.abs(self.audio_data))
        if max_audio > 0:
            self.audio_data = self.audio_data / max_audio
        print("Audio Normalized")


    def save_audio(self, output_filepath):
        """
        Saves the processed signal to a new WAV filepath
        """
        wavfile.write(output_filepath, self.sample_rate, self.audio_data)
    
    def process(self):
        self.sample_rate, self.audio_data = self.get_sampling_rate(self.audio)

        #main processing function that calls defined functions
        self.mono_stereo()
        self.normalize_audio()
        #self.play_sound()
        self.save_audio('original.wav')
        self.resample_audio()
        time, cosine_signal = self.generate_cos()
        self.plot_waveform()
        self.plot_cos(time, cosine_signal)
        #self.play_sound()
        self.save_audio('converted.wav')

    def create_bandpass(self, low_freq, high_freq, order):
        """
        Can also use scipy.signal.buttord to dynamically select order based on minimum requirements.
        :param low_freq:
        :param high_freq:
        :param order:
        :return:
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        return scipy.signal.butter(order, Wn=[low, high], btype='bandpass', fs=self.sample_rate, output='sos')



if __name__ == "__main__":
    audio = '03-laufey-from-the-start.wav'
    processor = SignalProcessor(audio)
    processor.process()
