import os
import sys
import numpy as np 
import pandas as pd
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


labels_file  = '../data/labels.csv'
labels = pd.read_csv(labels_file,header=0)
img_dir = '../spectrogram_images_299/'

def extract_images():
    for path in labels['path']:
        try:
            # Read au-file
            print(path)
            y, sr = lb.load(path, sr=22050)  # Use the default sampling rate of 22,050 Hz

            # Compute spectrogram
            M = lb.feature.melspectrogram(y, sr,
                                               fmax=sr / 2,  # Maximum frequency to be used on the on the MEL scale
                                               n_fft=2048,
                                               hop_length=512,
                                               n_mels=96,  # Set as per the Google Large-scale audio CNN paper
                                               power=2)  # Power = 2 refers to squared amplitude

            # Power in DB
            log_power = lb.power_to_db(M, ref=np.max)  # Covert to dB (log) scale

            # Plotting the spectrogram
            fig = plt.figure(figsize=(5, 5))
            plt.axis('off')
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            librosa.display.specshow(log_power, cmap=cm.jet)
            fig.add_axes(ax)
            #fig.patch.set_visible(False)
            newpath = path.split("/")[4]
            fig.savefig(img_dir + newpath[0:-3] + '.png', frameon=True,bbox_inches=None, pad_inches=0.0)
            plt.cla()
            plt.clf()
            plt.close()

        except Exception as e:
            print(path, e)
            pass
