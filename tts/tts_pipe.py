from .inference import VITS
import numpy as np



class TTSpeech(VITS):
    def __init__(self, tts_config):
        self.config = tts_config
        super(TTSpeech, self).__init__(pth=self.config["tts_path"], config=self.config["kss_path"])
        self.demo = self.config["demo"]
    
    def predict(self, text):
        sr, audio = super(TTSpeech, self).generate(text)
        return sr, audio