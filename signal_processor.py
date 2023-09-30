import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html

class SignalProcessor():
    def __init__(self, audio):
        self.audio = audio

    def get_sampling_rate(self):
        """
        The function determines sampling rate in kHz of an input signal.
        :return: sampling rate
        """
        return

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
    test_audio = None
    processor = SignalProcessor(test_audio)
