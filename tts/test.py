from class.SpeechVQAPipeline.tts.inference2 import VITS
import soundfile as sf
import time
import torchaudio
from pydub import AudioSegment
import IPython.display as ipd
import librosa


kss = VITS('./kss_second/D_98000.pth', config='./configs/kss.json')
myvits = kss

st = time.time()
audio = myvits.generate('횡단보도에 차가 있습니다.')
# audio = ipd.Audio(audio, rate=48000, normalize=True)
# xx = torch.Tensor(audio)
et = time.time()
print(f"generation time : {et-st}")

from scipy.io.wavfile import write
write("example.wav", 48000, audio)
# sf.write('stereo_file1.wav', audio, 48000, 'PCM_24')
# print(audio)
# sf.write('example.wav', audio, 2*len(audio))
# torchaudio.save('example_TTS_input_text.wav', audio.view([1,-1]), 48000)
# audio = AudioSegment.from_file(audio)
# audio.export('examples.wav', format='wav')

# audio.export(audio, format="wav", bitrate="64k")
# with open('test.wav', 'wb') as f:
#     f.write(audio.data)