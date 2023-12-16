from asr import ASRecognizer
from beit3 import VQAnswering
from tts import TTSpeech
import yaml
import gradio as gr


class SVQA():
    def __init__(self, config:dict, demo:bool=True):
        self.config = config

        asr_config = config["asr"]
        beit3_config = config["beit3"]
        tts_config = config["tts"]
        if not demo:
            asr_config["demo"] = False
            beit3_config["demo"] = False
            tts_config["demo"] = False
        
        # Model Initialization
        self.asr_pipe = ASRecognizer(asr_config)
        self.beit3_pipe = VQAnswering(beit3_config) 
        self.tts_pipe = TTSpeech(tts_config)
    
    def speech_recognition(self, sound):
        output = self.asr_pipe.predict(sound)
        return output
    
    def vq_answering(self, text, image):
        output = self.beit3_pipe.predict(text, image)
        return output
    
    def ttspeech(self, text):
        output = self.tts_pipe.predict(text)
        return output
   
    def run(self, sound, image):
        output = self.speech_recognition(sound)
        print("### Speech Recognized ###")
        print(f"### question : {output} ###")
        output = self.vq_answering(output, image)
        print("### VQA Answered ###")
        print(f"### answer : {output} ###")
        sr, output = self.ttspeech(output)
        return sr, output
    
        
if __name__ == "__main__":
    with open('config.yml', 'r') as f:
        config = yaml.full_load(f)
    
    vqa_engine = SVQA(config=config)

    demo = gr.Interface(vqa_engine.run,
                        inputs = [gr.Audio(type='filepath'), #sources=["microphone"]
                                 gr.Image(type="pil")],
                        outputs = "audio")
    demo.launch(share=True)
        