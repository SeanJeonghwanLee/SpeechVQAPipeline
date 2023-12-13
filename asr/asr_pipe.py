from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import time


class ASRecognizer():
    def __init__(self, asr_path:str, demo:bool=False):
        self.demo = demo
        # load model and processor
        self.asr_path = asr_path
        self.processor = WhisperProcessor.from_pretrained(self.asr_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.asr_path)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="korean", task="transcribe")

    def predict(self, sound):
        if self.demo:
            sampling_rate, data = sound
        else: 
            data = sound
            sampling_rate = 48000
        
        input_features = self.processor(data, sampling_rate=sampling_rate, return_tensors="pt").input_features
        predicted_ids = self.model.generate(input_features, forced_decoder_ids = self.forced_decoder_ids)
        result = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return result
