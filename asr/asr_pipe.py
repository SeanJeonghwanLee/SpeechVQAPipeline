from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import numpy as np
import librosa


class ASRecognizer():
    def __init__(self, asr_config:dict):
        self.config = asr_config
        self.demo = self.config["demo"]
        # load model and processor
        self.asr_path = self.config['model']
        self.processor = WhisperProcessor.from_pretrained(self.asr_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.asr_path)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="korean", task="transcribe")

    def predict(self, sound):
        if self.demo:
            data, origin_sr = librosa.load(sound)
        else:
            data, origin_sr = sound
        
        if origin_sr != 16000:
            data = self._down_sample(data, origin_sr, resample_sr=16000)
        
        input_features = self.processor(data, sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = self.model.generate(input_features, forced_decoder_ids = self.forced_decoder_ids)
        result = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return result

    def _down_sample(self, input_wav, origin_sr, resample_sr):
        resample = librosa.resample(input_wav, orig_sr = origin_sr, target_sr = resample_sr)
        return resample



if __name__ == "__main__":
    import yaml
    import time

    with open('../config.yml', 'r') as f:
        config = yaml.full_load(f)
    asr_config = config["asr"]
    asr_config["demo"] = False
    print("###config loaded###")

    asr = ASRecognizer(asr_config)
    print("###Model loaded###")

    # sample audio
    audio = librosa.load("./sample/New Recording 25.wav", sr=16000)


    st = time.time()
    result = asr.predict(audio)
    et = time.time()
    print(f"result ({et-st} sec): {result}")