from asr import ASRecognizer
from beit3 import beit3_CLASS
from tts import tts_CLASS
import yaml
import gradio as gr


class SVQA():
    def __init__(self, config, demo=True):
        self.config = config
        asr_path = config["asr_path"]
        beit3_path = config["beit3_path"]
        tts_path = config["tts_path"]
        
        # Model Initialization
        self.asr_pipe = ASRecognizer(asr_path, demo)
        self.beit3_pipe = beit3_CLASS(beit3_path, demo) 
        self.tts_pipe = tts_CLASS(tts_path, demo)
    
    def speech_recognition(self, sound):
        output = self.asr_pipe.predict(sound)
        return output
    
    def vq_answering(self, text):
        output = self.beit3_pipe.predict(text)
        return output
    
    def ttspeech(self, text):
        output = self.tts_pipe.predict(text)
        return output
   
    def run(self, sound):
        output = self.speech_recognition(sound)
        output = self.vq_answering(output)
        output = self.ttspeech(output)
        return output
    
        
if __name__ == "__main__":
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    
    vqa_engine = SVQA(config=config)

    demo = gr.Interface(vqa_engine.run,
                        gr.Audio(sources=["microphone"]),
                        output="audio")
    demo.launch()
        