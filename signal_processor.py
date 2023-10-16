import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
from scipy.signal import resample



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
    
    def mono_stereo(self, audio):
        """
        checks whether input sound is stereo or mono 
        if stereo, add both columns to a signal channel (1 column array)
        """
        if len(self.audio_data.shape) == 2:
            self.audio_data = self.audio_data.sum(axis=1)/2 # converts to mono audio from stereo

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
        return
    def save_audio(solf, output_filepath):
        """
        Saves the processed signal to a new WAV filepath
        """
        wavfile.write(output_filepath, self.sample_rate, self.audio)



if __name__ == "__main__":
    test_audio = None
    processor = SignalProcessor(test_audio)
