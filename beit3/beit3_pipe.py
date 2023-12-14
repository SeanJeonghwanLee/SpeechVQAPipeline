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

import utils




class VQAFactory():
    def __init__(self, beit3_config:dict):
        self.config = beit3_config
        self.demo = self.config["demo"]
        self.device = torch.device(self.config["device"])
        self._fix_seed()
        self.init_distributed_mode()
        cudnn.benchmark = True
        
        #tokenizer initialize
        self.tokenizer = XLMRobertaTokenizer(self.config["sentencepice_model"])
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.label2ans = self._label2ans()
        
        
        self.input_size = self.config["input_size"]
        self.transform = self._build_transform()
        
        
        #model initialize
        model_config = self.config["model"]
        self.model = create_model(
            model_config,
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
        answer = []
        # tokenization and transform
        language_tokens, padding_mask, _ = self._tokenize(text)
        image = self._transform(image)
        
        logits = self.model(
            image=image, question=language_tokens, 
            padding_mask=padding_mask)
        _, preds = logits.max(-1)
        for pred in preds:
            answer.append({
                "answer": self.label2ans[pred.item()], 
            })
        
        return answer


    def _tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        max_len = 64
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        language_tokens = tokens + [self.pad_token_id] * (max_len - num_tokens)

        return language_tokens, padding_mask, num_tokens
    
    
    def _transform(self, image):
        return self.transform(image)

    def _fix_seed(self, seed:int=666666):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _lable2ans(self, ans2label):
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
        # elif 'SLURM_PROCID' in os.environ:
        #     args.rank = int(os.environ['SLURM_PROCID'])
        #     args.gpu = args.rank % torch.cuda.device_count()

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



if __name__ == '__main__':
    pass # pipe test required