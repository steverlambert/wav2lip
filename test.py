import numpy as np
import audios as aud
#a,b = librosa.load('audio/hello.wav', sr=16000, res_type='scipy')
a = aud.load_wav('audio/hello.wav', sr=16000, rosa=True)
print(np.count_nonzero(a), a)

print(aud.melspectrogram(a))