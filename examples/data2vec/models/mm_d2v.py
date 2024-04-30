import torch.nn as nn
import torch
import numpy as np
from functools import partial
import logging
from typing import *
import torch.nn.functional as F
import math
from torchvision.transforms import v2 as transforms
import pytorch_lightning as L

from examples.data2vec.data.modality import Modality
from examples.data2vec.models.modalities.modules import AltBlock
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from transformers.optimization import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)

def get_transforms(no_transform=False, beit_transforms=False,
                   transform_jitter=False, crop_scale=(0.08, 1.0)):
    
    transform_prepare = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
            ]
    )

    if transform_jitter:
        transform_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    if no_transform:
        transform_train = transforms.Resize((224, 224))
    elif beit_transforms:
        beit_transform_list = []
        beit_transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.4))
        beit_transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        transform_train = transforms.Compose(beit_transform_list)
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=(224, 224), scale=crop_scale, interpolation=3
                ),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
            ]
        )
    final_transform = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    if transform_jitter:
        return transforms.Compose(
            [
                transform_prepare,
                transform_train,
                transform_jitter,
                final_transform,
            ]
        )
    else:
        return transforms.Compose([transform_prepare, transform_train, final_transform])



class KDData2VecPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = KDMMData2Vec(cfg=self.cfg.model)
        self.transform = get_transforms(**cfg.image_transforms)
        
        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx):
        target:torch.Tensor = batch.pop('target')
        if batch['modes'][0] == Modality.IMAGE:
            batch['image'] = self.transform(batch['image'])
        
        output_dict = self(batch) # call "forward"

        if batch['modes'][0] == Modality.AUDIO:
            y_hat = average_twice(output_dict['layer_results'], norm=True)

        else:
            #y_hat = special_token_and_average(output_dict['layer_results'], norm=True)
            y_hat = special_tokens(output_dict['layer_results']) # B, L, C

            target = target[:, -self.cfg.model.depth:, :] # only take the last layers so that the target has the same shape as y_hat

        assert y_hat.shape == target.shape # this must be the case

        loss = self.kd_loss(y_hat=y_hat, y=target)
        self.log("train/loss", loss, prog_bar=True)
        if batch['modes'][0] == Modality.IMAGE:
            self.log("train/loss_img", loss, prog_bar=True)
        elif batch['modes'][0] == Modality.AUDIO:
            self.log("train/loss_audio", loss, prog_bar=True)
        else:
            self.log("train/loss_text", loss, prog_bar=True)
        return loss
                
    
    def kd_loss(self, y_hat, y):
        y_hat = y_hat.view(-1, y_hat.size(-1)).float() # (B, D, C) -> (B*D, C)
        y = y.contiguous()
        y = y.view(-1, y.size(-1)).float() # (B, D, C) -> (B*D, C)

        loss = F.mse_loss(y_hat, y, reduction="none").float()

        if self.cfg.model.loss_scale is not None:
            scale = self.cfg.model.loss_scale
        else:
            scale = 1 / math.sqrt(y_hat.size(-1))
        
        reg_loss = loss * scale
        
        return reg_loss.sum(dim=-1).mean() # sum over the last dimension and then take the mean over the batch


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr,
                                      betas=tuple(self.cfg.optimizer.betas),
                                      eps=self.cfg.optimizer.eps,
                                      weight_decay=self.cfg.optimizer.weight_decay)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
            num_training_steps=self.cfg.optimizer_schedule.max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "cosine_w_warmup"}]

class KDMMData2Vec(nn.Module):
    def __init__(self,
                 cfg,
                 ):
        super(KDMMData2Vec, self).__init__()
        self.cfg = cfg
        self.supported_modalities = cfg.supported_modalities

        self.projections = nn.ModuleDict({
            mode.name.lower(): 
            (nn.Linear(self.cfg.encoders_embed_dim, self.cfg.embed_dim) 
             if self.cfg.modality_encoder_proj 
             else nn.Identity())
             for mode in self.supported_modalities
        })

        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)

        self.blocks = nn.ModuleList([self.make_block(dpr[i]) for i in range(self.cfg.depth)])

        self.norm = None
        if self.cfg.layer_norm_first:
            self.norm = make_layer_norm(self.cfg.embed_dim)

        self.layerdrop = self.cfg.layerdrop
        self.mask_seed = self.cfg.seed

        # self.apply(self._init_except_pretrained)
        self.apply(init_bert_params)

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        # add modality encoders later, so that they are not part of the model's parameters when model is initialized
        self.modality_encoders:nn.ModuleDict[Modality, nn.Module] = self._get_modality_encoders()
        self._freeze(self.modality_encoders)
        self.modality_encoders.eval()
        
    def make_block(self, drop_path, dim=None, heads=None):
        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        return AltBlock(
            dim=self.cfg.embed_dim if dim is None else dim,
            num_heads=self.cfg.num_heads if heads is None else heads,
            mlp_ratio=self.cfg.mlp_ratio,
            qkv_bias=True,
            drop=self.cfg.encoder_dropout,
            attn_drop=self.cfg.attention_dropout,
            mlp_drop=self.cfg.activation_dropout,
            post_mlp_drop=self.cfg.post_mlp_drop,
            drop_path=drop_path,
            norm_layer=make_layer_norm,
            layer_norm_first=self.cfg.layer_norm_first,
            ffn_targets=not self.cfg.end_of_block_targets,
        )


    def _init_except_pretrained(self, module:nn.Module):
        if all(param.requires_grad for param in module.parameters(recurse=False)):
            init_bert_params(module)

    def forward(
        self,
        modes:List[Modality],
        audio:torch.Tensor=None,
        image:torch.Tensor=None,
        text:torch.Tensor=None,
        id:torch.Tensor=None,
        padding_mask:torch.Tensor=None, 
        mask:bool=False,
        features_only:bool=True,
        force_remove_masked:bool=False,
        remove_extra_tokens:bool=False,
        precomputed_mask=None,
    ):
        assert len(modes)==1, f"This model accepts exactly modality indicator at a time, received: {modes}"
        n_sources = sum(mode is not None for mode in [audio, image, text])
        assert n_sources==1,\
            f"This model accepts exactly one modality data source at a time, got {n_sources}."
        
        assert modes[0] in self.supported_modalities, f"Unsupported modality: {modes[0]}, supported modalities are: {self.supported_modalities}"
        mode = modes[0].name.lower() # is now a string, ModuleDict does not support enums as keys
        
        if audio is not None:
            source = audio
        elif image is not None:
            source = image
        elif text is not None:
            source = text
        else:
            raise ValueError("Audio, image or text must be provided, found all to be None.")
        
        feature_extractor = self.modality_encoders[mode]
        with torch.no_grad():
            extractor_out = feature_extractor(
                source,
                padding_mask,
                mask,
                remove_masked=not features_only or force_remove_masked,
                clone_batch=1,
                mask_seeds=None,
                precomputed_mask=precomputed_mask,
            )

        extractor_out = self.projections[mode](extractor_out)

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.layerdrop == 0
                or (np.random.random() > self.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        if remove_extra_tokens:
            x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
            if masked_padding_mask is not None:
                masked_padding_mask = masked_padding_mask[
                    :, feature_extractor.modality_cfg.num_extra_tokens :
                ]

        return {
            "x": x,
            "padding_mask": masked_padding_mask,
            "layer_results": layer_results,
            "mask": encoder_mask,
        }
    
    def extract_features(
        self, audio=None, image=None, text=None, modes:List[Modality]=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        res = self.forward(
            audio=audio,
            image=image,
            text=text,
            modes=modes,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        return res
    
    @torch.no_grad()
    def encode_modality(self, modes:Union[Modality, List[Modality]], source:torch.Tensor, padding_mask=None, normalize:bool=True):
        if isinstance(modes, List):
            assert len(modes)==1, 'Only one modality allowed when calling "encode_modality".'
            mode = modes[0]
        # at this point Modality has to be of type "Modality".
        audio = None
        image = None
        text = None
        if mode == Modality.AUDIO:
            audio = source
        elif mode == Modality.IMAGE:
            image = source
        elif mode == Modality.TEXT:
            text = source
        else:
            raise ValueError(f"Did not find mode for modality \"{mode}\", allowed modes are: [{Modality.AUDIO}, {Modality.IMAGE}, {Modality.TEXT}]")

        output = self.extract_features(
            audio=audio,
            image=image,
            text=text,
            modes=[mode],
            padding_mask=padding_mask,
            remove_extra_tokens=False, # important!
        )['x']

        if mode == Modality.AUDIO:
            output = output.mean(dim=1)
        else:
            output = output[:, 0, :]

        if normalize:
            output = F.normalize(output, dim=-1)
        
        return output # shape: (batch_size, embed_dim)

    def encode_audio(self, audio, padding_mask, normalize:bool=True):
        return self.encode_modality(source=audio,
                                    modes=Modality.AUDIO,
                                    padding_mask=padding_mask,
                                    normalize=normalize)
    
    def encode_image(self, image, normalize:bool=True):
        return self.encode_modality(source=image,
                                    modes=Modality.IMAGE,
                                    normalize=normalize)

    def encode_text(self, text, padding_mask, normalize:bool=True):
        return self.encode_modality(source=text,
                                    modes=Modality.TEXT,
                                    padding_mask=padding_mask,
                                    normalize=normalize)

    def _get_pretrained_block_indices(self, depth, n_blocks_pretrained) -> List[int]:
        blocks_pretrained = [i for i in range(n_blocks_pretrained)]
        blocks = []
        pretrained_blocks_count = len(blocks_pretrained)

        if depth*2 > pretrained_blocks_count:
            pretrained_remaining = pretrained_blocks_count
            model_remaining = depth
            current_block_idx = 0
            while model_remaining*2 > pretrained_remaining and model_remaining > 0:
                blocks.append(blocks_pretrained[current_block_idx])
                pretrained_remaining -= 1
                model_remaining -= 1
                current_block_idx += 1
            
            if model_remaining > 0:
                for i in range(0, depth-current_block_idx):
                    take_block_idx = i*2+current_block_idx
                    blocks.append(blocks_pretrained[take_block_idx])
        else:
            for i in range(depth):
                blocks.append(blocks_pretrained[i*2])
        
        assert len(blocks) == depth

        return blocks
    
    def freeze_attention_blocks(self):
        if self.cfg.freeze_attention:
            for block in self.blocks:
                self._freeze(block.attn)
                block.attn.eval()

    def unfreeze_attention_blocks(self):
        if self.cfg.freeze_attention:
            for block in self.blocks:
                self._unfreeze(block.attn)
                block.attn.train()

    def _freeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = True

    def remove_modalities_except(self, keep_modes:List[Modality]) -> None:
        """
        Removes all modalities from the model except the ones specified.
        Useful when fine-tuning the model on downstream task
        involving only a subset of the supported modalities.
        """
        for modality in self.supported_modalities:
            if modality not in keep_modes:
                del self.projections[modality.name.lower()]
                del self.modality_encoders[modality.name.lower()]
