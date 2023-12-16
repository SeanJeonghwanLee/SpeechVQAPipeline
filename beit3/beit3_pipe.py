import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path


from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
from transformers import XLMRobertaTokenizer

from . import utils
from . import modeling_finetune



class VQAnswering():
    def __init__(self, beit3_config:dict):
        self.config = beit3_config
        self.demo = self.config["demo"]
        self.device = torch.device(self.config["device"])
        self._fix_seed()
        self.init_distributed_mode()
        cudnn.benchmark = True
        
        #tokenizer initialize
        self.tokenizer = XLMRobertaTokenizer(self.config["sentencepiece_model"])
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.label2ans = self._label2ans(self.config["ans2label"])
        
        
        self.input_size = self.config["image_size"]
        self.transform = self._build_transform()
        
        
        #model initialize
        _model = self.config["model"]
        self.model = create_model(
            _model,
            pretrained=False,
            drop_path_rate=self.config["drop_path"],
            vocab_size=self.config["vocab_size"],
            checkpoint_activations=self.config["checkpoint_activations"]
        )

        utils.load_model_and_may_interpolate(self.config["model_path"], self.model, self.config["model_key"], self.config["model_prefix"])
        self.model.to(self.device)

        torch.distributed.barrier()
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu], find_unused_parameters=True)
        self.model._set_static_graph()


    def predict(self, text, image):
        # tokenization and transform
        language_tokens, padding_mask, _ = self._tokenize(text)
        image = self._transform(image)
        
        logits = self.model(
            image=image, question=language_tokens, 
            padding_mask=padding_mask)
        _, preds = logits.max(-1)
        answer = self.label2ans[preds[0].item()]
                
        return answer


    def _tokenize(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        max_len = 64
        if len(token_ids) > max_len - 2:
            token_ids = token_ids[:max_len - 2]

        token_ids = [self.bos_token_id] + token_ids[:] + [self.eos_token_id]
        num_tokens = len(token_ids)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        language_tokens = token_ids + [self.pad_token_id] * (max_len - num_tokens)

        language_tokens = torch.Tensor(language_tokens).type('torch.LongTensor').reshape(1,-1)
        padding_mask = torch.Tensor(padding_mask).type('torch.LongTensor').reshape(1,-1)
        return language_tokens, padding_mask, num_tokens
    
    
    def _transform(self, image):
        image = self.transform(image)
        C, H, W = image.shape
        image = image.reshape(1, C, H, W)
        return image

    def _fix_seed(self, seed:int=666666):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _label2ans(self, ans2label):
        label2ans = []
        with open(ans2label, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                ans = data["answer"]
                label2ans.append(ans)
        return label2ans
        
    def init_distributed_mode(self):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            self.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['WORLD_SIZE'])
            self.gpu = rank % torch.cuda.device_count()

        # args.distributed = True
        dist_backend = self.config["dist_backend"]
        dist_url = self.config["dist_url"]

        torch.cuda.set_device(self.gpu)
        torch.distributed.init_process_group(
            backend=dist_backend, init_method=dist_url,
            world_size=world_size, rank=rank,
            timeout=datetime.timedelta(0, 7200)
        )
        torch.distributed.barrier()
        # setup_for_distributed(args.rank == 0)
    
    def _build_transform(self):
        t = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
        return t



if __name__ == "__main__":
    from torchvision.datasets.folder import default_loader
    
    import yaml
    from PIL import Image
    import time


    with open('../config.yml', 'r') as f:
        config = yaml.full_load(f)
    beit3_config = config["beit3"]
    beit3_config["demo"] = False
    print("###config loaded###")

    vqa = VQAnswering(beit3_config)
    print("###Model loaded###")


    # sample
    image = default_loader("./sample/three_puppies.jpg")
    # text = "무슨 동물인가요?"
    text = "강아지가 몇 마리인가요?"

    st = time.time()
    result = vqa.predict(text, image)
    et = time.time()
    print(f"result ({et-st} sec): {result}")