import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
from scipy.signal import resample


class SignalProcessor():
    def __init__(self):
        self.sample_rate = None
        self.audio = None

    def read_audio(self, filepath):
        """
        :param filepath: relative path to the audio file
        :return: None
        """
        self.sample_rate, self.audio = wavfile.read(filepath)

    def get_sampling_rate(self):
        """
        3.6
        The function determines sampling rate in kHz of an input signal.
        :return: sampling rate
        """
        if self.sample_rate:
            print(f"Sampling Rate of Original WAV file: {self.sample_rate} Hz")
        else:
            print("You must upload an audio before calculating sampling rate")

        return self.sample_rate

    def play_sound(self):
        """
        Plays the sound out loud.
        :return: None
        """
        return

    def plot_waveform(self):
        """
        Plots the sound waveform as a function of the sample number
        :return:
        """
        return


if __name__ == "__main__":

    # testing methods
    test_audio1 = 'audio-files/02-coffee-shop-ambiance.wav'
    processor = SignalProcessor()
    processor.read_audio(test_audio1)
    processor.get_sampling_rate()

    # more tests
    test_audio2 = 'audio-files/01-pickle-asmr-quiet.wav'
    processor = SignalProcessor()
    processor.read_audio(test_audio2)
    processor.get_sampling_rate()

