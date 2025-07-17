from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL, NegNCE
import numpy as np
import copy
allgather = AllGather.apply
allgather2 = AllGather2.apply

logger = logging.getLogger(__name__)

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class VTRModel(nn.Module):
    def __init__(self, config):
        super(VTRModel, self).__init__()
        
        self.config = config
        backbone = getattr(config, 'base_encoder', "ViT-B/32")
        self.video_frames = getattr(config, "max_frames", None)

        self.lora_dim = config.lora_dim
        logger.info("v_LoRA: {} dim".format(self.lora_dim))
        
        assert backbone in _PT_NAME
        model_path = os.path.join(config.pretrained_path, _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size 

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, self.lora_dim, 
                         self.video_frames, config)
            
        self.loss_fct = CrossEn(config)
        self.loss_neg = NegNCE()
        self.clip.load_state_dict(state_dict, strict=False)
        
    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):

        batch_size = video.size(0)
        
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video_frame = n_v
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video_frame = bs
            video = video.view(b * pair * bs * ts, channel, h, w)

        cls = self.get_text_feat(text_ids, text_mask)
        video_feat = self.get_video_feat(video, video_mask, video_frame=video_frame)
        
        cls = allgather(cls, self.config)
        video_feat = allgather(video_feat, self.config)
        torch.distributed.barrier()
        
        logit_scale = self.clip.logit_scale.exp()
        loss = 0.
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True)
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])

        loss_t2v = self.loss_fct(t2v_logits * logit_scale)
        loss_v2t = self.loss_fct(t2v_logits.T * logit_scale)
        loss = (loss_t2v + loss_v2t) / 2

        return loss

    def stage0_eval(self, text_ids, text_mask, video, video_mask=None):

        batch_size = video.size(0)

        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video_frame = n_v
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video_frame = bs
            video = video.view(b * pair * bs * ts, channel, h, w)
        return text_ids, text_mask, video, video_mask, video_frame

    def stage1_eval(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids, text_mask, video, video_mask, video_frame = self.stage0_eval(
                        text_ids, text_mask, video, video_mask)

        cls = self.get_text_feat(text_ids, text_mask)
        video = self.get_video_feat(video, video_mask, video_frame=video_frame)
        
        return cls, video

    def stage2_eval(self, cls, text_mask, video_feat, video_mask):
        logit_scale = self.clip.logit_scale.exp()
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True) 
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True) 

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])
        
        return t2v_logits * logit_scale

    def get_text_feat(self, text_ids, orig_mask):
        b = text_ids.size(0)
        x = self.clip.token_embedding(text_ids)
        max_t_len = x.size(1)
        pos_emd = self.clip.positional_embedding[:max_t_len, :]
        x = x + pos_emd

        mask = orig_mask
        text_length = max_t_len
        attn_mask = self.clip.build_attention_mask(text_length).repeat(x.size(0), 1, 1).to(mask.device)
        inf = torch.zeros((text_length, text_length)).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)

        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        attn_mask = torch.where(mask>0, attn_mask, inf)
    
        x = self.clip.transformer(x, attn_mask)

        hidden = self.clip.ln_final(x) @ self.clip.text_projection
        cls = hidden[torch.arange(hidden.shape[0]), text_ids.argmax(dim=-1)]

        cls = cls.float()
        cls = cls.view(b, -1, cls.size(-1)).squeeze(1)
        return cls

    def get_video_feat(self, video, video_mask, video_frame=-1):
        """image enconding"""

        video_feat = self.clip.encode_image(video, video_frame=video_frame)
        video_feat = video_feat.float().squeeze(1).contiguous()
        
        return video_feat

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
