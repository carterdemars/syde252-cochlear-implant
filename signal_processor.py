import scipy
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd


class SignalProcessor():
    def __init__(self, audio):
        self.audio = audio
        self.audio_data = None  # Data is read from audio file and stored here
        self.sample_rate = None  # Sampling rate of the audio file
        self.channels = []
        self.envelopes = []

    def get_sampling_rate(self, audio):
        """
        3.1: reads the file first, then finds sampling rate
        The function determines sampling rate in kHz of an input signal.
        :return: sampling rate
        """
        self.sample_rate, self.audio = wavfile.read(audio)
        print(f"Sampling Rate of Original WAV file: {self.sample_rate} Hz")
        return self.sample_rate, self.audio

    def mono_stereo(self):
        """
        checks whether input sound is stereo or mono 
        if stereo, add both columns to a signal channel (1 column array)
        """
        if len(self.audio_data.shape) == 2:
            self.audio_data = self.audio_data.sum(axis=1) / 2  # converts to mono audio from stereo
        print("Coverted from Stereo to Mono")

    def play_sound(self):
        """
        Plays the sound out loud.
        :return: None
        """
        sd.play(self.audio_data, self.sample_rate)  # plays data extracted from audio file at found sampling rate
        sd.wait()  # blocks python interpreter until playback is finished
        return

    def plot_waveform(self):
        """
        Plots the sound waveform as a function of the sample number
        :return:
        """
        plt.figure(figsize=(10, 4))
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
            self.audio_data = signal.resample(self.audio_data, int(16000 / self.sample_rate * len(self.audio_data)))
            self.sample_rate = 16000
            print("Audio Resampled")

    def generate_cos(self):
        """
        Generates a cosine signal of 1kHz frequency that has the same duration as the audio signal
        """
        # duration of signal (total time in s that original audio lasts) = total number of samples/sample rate
        duration = len(self.audio_data) / self.sample_rate

        # creating linearly spaced array for 1kHz
        time = np.linspace(0., duration, len(self.audio_data))

        frequency = 1000  # 1kHz

        # generating cosine wave: A * cos(2*pi*frequency*t)
        # where A = amplitude, t = time
        cosine_signal = np.cos(2 * np.pi * frequency * time)

        return time, cosine_signal

    def plot_cos(self, time, cosine_signal):
        """
        Plots the first two cycles of the generated cos signal
        """
        freq = 1000
        plt.figure(figsize=(10, 4))
        # plotting for two cycles:
        # x axis: divide sample rate by frequency, and multiply by 2 to plot 2 cycles
        # y axis: same as above, but with cosine signal
        two_cycles = int((2 / freq) * self.sample_rate)
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

        # main processing function that calls defined functions
        self.mono_stereo()
        self.normalize_audio()
        # self.play_sound()
        self.save_audio('original.wav')
        self.resample_audio()
        time, cosine_signal = self.generate_cos()
        self.plot_waveform()
        self.plot_cos(time, cosine_signal)
        # self.play_sound()
        self.save_audio('converted.wav')

    def create_bandpass_filters(self, N, low_freq=100, high_freq=8000):
        self.bandpass_filters = []
        self.filter_freq_ranges = []  # Separate list for frequency ranges
        freq_bands = np.logspace(np.log10(low_freq), np.log10(high_freq), N + 1)
        for i in range(N):
            bp_filter = signal.butter(8, Wn=[freq_bands[i], freq_bands[i + 1]], btype='bandpass', fs=self.sample_rate, output='sos')
            self.bandpass_filters.append(bp_filter)  # Store only filter coefficients
            self.filter_freq_ranges.append((freq_bands[i], freq_bands[i + 1]))  # Store frequency ranges separately

    def apply_filters(self):
        filtered_signals = []
        for bp_filter in self.bandpass_filters:
            filtered_signal = signal.sosfilt(bp_filter, self.audio_data)
            filtered_signals.append(filtered_signal)
        return filtered_signals


    def plot_filtered_signals(self, filtered_signals):
        """
        Plots the output signals of the lowest and highest frequency channels on subplots
        """
        plt.figure(figsize=(10, 8))

        # Plot for the lowest frequency channel
        plt.subplot(2, 1, 1)
        plt.plot(filtered_signals[0])
        plt.title('Lowest Frequency Channel')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        # Plot for the highest frequency channel
        plt.subplot(2, 1, 2)
        plt.plot(filtered_signals[-1], color='red')
        plt.title('Highest Frequency Channel')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

    def envelope_extraction(self, cutoff_frequency=400, order=8):

        # generate a lowpass filter with 400Hz cutoff
        b, a = scipy.signal.butter(order, cutoff_frequency, btype='low', analog=False, fs=self.sample_rate)

        # rectify the signal and apply the filter
        self.envelopes = [signal.filtfilt(b, a, np.abs(channel)) for channel in self.channels]

    def process2(self):
        N = 10
        self.sample_rate, self.audio_data = self.get_sampling_rate(self.audio)
        self.mono_stereo()
        self.normalize_audio()
        bandpass_filters = self.create_bandpass_filters(N)
        self.channels = self.apply_filters(bandpass_filters)
        self.plot_filtered_signals(self.channels)
        self.envelope_extraction()
        self.plot_filtered_signals(self.envelopes)
    
    def generate_cos_for_channel(self, frequency, length):
        duration = length / self.sample_rate
        time = np.linspace(0., duration, length)
        cosine_signal = np.cos(2 * np.pi * frequency * time)
        return cosine_signal
    
    def generate_cos_signals_for_all_channels(self):
        cosine_signals = []
        for (low_freq, high_freq) in self.filter_freq_ranges:
            central_freq = np.sqrt(low_freq * high_freq)
            length = len(self.envelopes[0])
            cosine_signal = self.generate_cos_for_channel(central_freq, length)
            cosine_signals.append(cosine_signal)
        return cosine_signals
    
    def amplitude_modulate(self, cosine_signal, rectified_signal):
        return cosine_signal * rectified_signal
    
    def play_and_save_output_audio(self, final_signal):
        sd.play(self.output_signal, self.sample_rate)
        sd.wait() 
        wavfile.write(final_signal, self.sample_rate, self.output_signal.astype(np.float32))
        print(f"Output audio saved to {final_signal}")


    
    def process3(self):
        N = 10
        self.sample_rate, self.audio_data = self.get_sampling_rate(self.audio)
        self.mono_stereo()
        self.normalize_audio()
        self.create_bandpass_filters(N)
        self.channels = self.apply_filters()
        self.plot_filtered_signals(self.channels)
        self.envelope_extraction()
        self.plot_filtered_signals(self.envelopes)
        #Task 10
        self.cosine_signals = self.generate_cos_signals_for_all_channels()
        #Task 11
        self.amp_modulated_signals = []
        #amplitude modulate for every signal
        for cosine_signal, rectified_signal in zip(self.cosine_signals, self.envelopes):
            amp_signal = self.amplitude_modulate(cosine_signal, rectified_signal)
            self.amp_modulated_signals.append(amp_signal)
        #Task 12
        #Add all amplitude modulated signals
        self.output_signal = np.sum(self.amp_modulated_signals, axis=0)
        #Normalize output signal
        #Task 13
        self.play_and_save_output_audio('final_output.wav')

        




if __name__ == "__main__":
    audio = '03-laufey-from-the-start.wav'
    processor = SignalProcessor(audio)
    processor.process3()
