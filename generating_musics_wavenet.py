import os

import matplotlib.pyplot as plt
import numpy as np
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen

# from https://freesound.org/people/MustardPlug/sounds/395058/
fname = './data/395058__mustardplug__breakbeat-hiphop-a4-4bar-96bpm.wav'
sr = 16000
audio = utils.load_audio(fname, sample_length=40000, sr=sr)
sample_length = audio.shape[0]
print(f'{sample_length} samples, {sample_length / sr} seconds')

encoding = fastgen.encode(
    audio, './data/wavenet-ckpt/model.ckpt-200000', sample_length)
print(encoding.shape)

np.save(fname + '.npy', encoding)
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
axs[0].plot(audio)
axs[0].set_title('Audio Signals')
axs[1].plot(encoding[0])
axs[1].set_title('Nsynth Encoding')
plt.show()

fastgen.synthesize(
    encoding,
    save_paths=[fname + '.gen.wav'],
    checkpoint_path='./data/wavenet-ckpt/model.ckpt-200000',
    samples_per_save=sample_length,
)
