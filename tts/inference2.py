import torch
import commons
from text import text_to_sequence
import utils
from models import SynthesizerTrn
from text.symbols import symbols
import IPython.display as ipd
from pydub import AudioSegment

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

class VITS:
    def __init__(self, pth, config='configs/kss.json'):
        self.hps = utils.get_hparams_from_file(config)
        self.net = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)
        self.noise_scale = 0.667
        self.noise_scale_w = 0.8
        self.length_scale = 1
        self.net.eval()
        utils.load_checkpoint(pth, self.net, None)
        
    def generate(self, text:str, display=False):
        stn_tst = get_text(text, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            audio = self.net.infer(x_tst, 
                                   x_tst_lengths, 
                                   noise_scale=self.noise_scale, 
                                   noise_scale_w=self.noise_scale_w, 
                                   length_scale=self.length_scale
                                   )[0][0,0].data.cpu().float().numpy()
        if display:
            ipd.display(ipd.Audio(audio, rate=self.hps.data.sampling_rate, normalize=True))
        
        return audio
