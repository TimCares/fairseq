import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import os
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from data2vec_fairseq.models.modalities.base import ModalitySpecificEncoder
from data2vec_fairseq.models.data2vec2 import Data2VecMultiModel, Data2VecMultiConfig
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import contextlib

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from data2vec_fairseq.models.modalities.modules import AltBlock

from omegaconf import OmegaConf, DictConfig
from collections import namedtuple
from fairseq.data import Dictionary
from fairseq.dataclass.utils import merge_with_parent

logger = logging.getLogger(__name__)

def load_model(pretrained_model_cfg:DictConfig,
               model_state_dict) -> Data2VecMultiModel:
    
    pretrained_model_cfg = merge_with_parent(Data2VecMultiConfig(), pretrained_model_cfg, remove_missing=True)

    logger.info(f"Modality used: {pretrained_model_cfg.supported_modality}")
    if pretrained_model_cfg.supported_modality.name.lower() == 'text':
        Task = namedtuple('Task', 'source_dictionary')

        dictionary = Dictionary.load(os.path.join('..', 'data', "dict.txt"))
        dictionary.add_symbol("<mask>")
        dummy_task = Task(source_dictionary=dictionary)
    else:
        dummy_task = False

    model = Data2VecMultiModel.build_model(pretrained_model_cfg, task=dummy_task)

    result = model.load_state_dict(model_state_dict)
    logger.info(f'Loaded state dict, result: {result}')
    return model


def load_pretrained_d2v_model(state_dict_path:str, keep_decoder:bool=False) -> Data2VecMultiModel:
    model_meta_data = torch.load('/Users/timcares/CompSci/Uni/Projects/MasterThesis/models/base_imagenet.pt')
    pretrained_model_cfg = OmegaConf.create(model_meta_data['cfg']['model'])
    model = load_model(pretrained_model_cfg=pretrained_model_cfg, model_state_dict=model_meta_data['model'])

    # removes decoder, and all encoders with modality != supported modality
    model.remove_pretraining_modules(modality=pretrained_model_cfg.supported_modality, keep_decoder=keep_decoder)

    return model

def prepare_output(out:List[torch.Tensor], modality:Modality, norm:bool=True) -> List[torch.Tensor]:
    if norm:
        out = [
            F.instance_norm(tl.transpose(1, 2).float()).transpose(1, 2)
            for tl in out  # BTC -> BCT -> BTC
        ]

    y = out[0].float()
    for tl in out[1:]:
        y.add_(tl.float())
    y = y.div_(len(out))

    if modality == Modality.IMAGE:
        y = F.layer_norm(y, y.shape[-1:])
    return y


def get_max_saliency_patches(
        frac_keep_tokens:float,
        attn_results:List[torch.Tensor],
        extractor_out:Dict[str, torch.Tensor],
        feature_extractor:ModalitySpecificEncoder,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
    alibi_scale = extractor_out.get("alibi_scale", None)
    x_unmasked = extractor_out['x_pre_context'] # will definitely be unmasked, as "d2v_masking" is False
    masked_padding_mask = extractor_out["padding_mask"]
    masked_alibi_bias = extractor_out.get("alibi_bias", None)
    
    # TODO: think about option for audio -> no special token at the beginning
    attn_score = attn_results[0].float()
    for tl in attn_results[1:]:
        attn_score.add_(tl.float())
    attn_score = attn_score.div_(len(attn_results))

    ### partly adapted from: https://arxiv.org/pdf/2302.10494 (MaskedKD) ###
    num_keep = int(frac_keep_tokens*attn_score.size(-1)) # or size(-2) because shape is (B, H, T, T)
    keep_timesteps = torch.topk(attn_score.mean(dim=1)[:, 0, 1:], num_keep).indices # attn_score.mean(dim=1): B x H x T x T -> B x T x T

    # pre context already contains special token at the beginning
    cls_save = x_unmasked[:, 0, :].unsqueeze(dim=1) # B x 1 x D
    x_unmasked = x_unmasked[:, 1:, :]

    embed_dim = x_unmasked.size(-1)
    index = keep_timesteps.unsqueeze(-1).repeat(1, 1, embed_dim)
    x_unmasked_tokens_only = torch.gather(x_unmasked, dim=1, index=index)
    x_unmasked_tokens_only = torch.cat((cls_save , x_unmasked_tokens_only), dim=1)
    # B x num_keep+1 x D -> one (+1) stems from additional special token

    ### end of adaptation ###

    if masked_padding_mask is not None:
        padding_masked_unmasked_tokens = torch.gather(masked_padding_mask[:, 1:], dim=1, index=keep_timesteps)
        # the following should never raise and exception, as long as "num_keep" is not larger than the number of
        # non-padding tokens in the input, this is because padded tokens have an attention score of 0, which is the minimum
        assert (~padding_masked_unmasked_tokens).all(), "All non-masked MaskedKD tokens should be padding tokens."

    # now compute modality encoder output for the teacher model, we mask the tokens before the
    # context encoder of the modality encoder -> same approach as d2v masking
    # we reuse the features before the context encoder, saved from the forward pass of the modality encoder at the beginning
    x_unmasked_tokens_only = feature_extractor.context_encoder(
        x_unmasked_tokens_only, # our "x" here
        masked_padding_mask, # masked padding mask can be reused from previous modality encoder forward pass
        masked_alibi_bias, # alibi can be reused from previous modality encoder forward pass
        alibi_scale[: feature_extractor.modality_cfg.prenet_depth]
        if alibi_scale is not None
        else None,
    )
    return x_unmasked_tokens_only, keep_timesteps

def prepare_salient_patches(
        layer_results:List[torch.Tensor],
        keep_timesteps:torch.Tensor,
        mode:Modality,
        norm_first:bool,
        ) -> torch.Tensor:
    # if final_attn_layer_saliency_score is True, then "layer_results" only contains one element, the last layer
    # ... but it can be treated the same way as if it contains all layers

    if norm_first: # norms over all time steps, including those that are not used for the teacher
        layer_results = [F.instance_norm(tl.transpose(1, 2).float()).transpose(1, 2) for tl in layer_results]
    
    layer_results = torch.stack(layer_results)
    cls_save = layer_results[:, :, 0, :].unsqueeze(dim=2) # L x B x 1 x D
    layer_results = layer_results[:, :, 1:, :]

    embed_dim = layer_results.size(-1)
    # add dim for embedding dimension (-1) and for layer dimension (0)
    n_layers = layer_results.size(0)
    index = keep_timesteps.unsqueeze(-1).unsqueeze(0).repeat(n_layers, 1, 1, embed_dim)
    layer_results = torch.gather(layer_results, dim=2, index=index)
    layer_results = torch.cat((cls_save , layer_results), dim=2)
    # "layer_results" now only consists of same tokens as "x_unmasked_tokens_only"

    # averaged and normed layer results only on unmasked tokens
    # -> teacher only gets the unmasked tokens, and the teacher output is normed,
    # so we need to norm only the unmasked tokens for the student output as well
    layer_results = [layer_results[i] for i in range(len(layer_results))] # expand to list
    layer_results = prepare_output(out=layer_results, modality=mode, norm=not norm_first)
    # B x num_keep+1 x D -> one (+1) stems from additional special token
    return layer_results


class KDData2VecPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.d2v_masking = self.cfg.model.mask and self.cfg.model.d2v_masking
        self.masked_kd = self.cfg.model.mask and not self.cfg.model.d2v_masking and not self.cfg.model.inverse_masked_kd
        self.inverse_masked_kd = self.cfg.model.mask and not self.cfg.model.d2v_masking and self.cfg.model.inverse_masked_kd

        state_dict_name = self.cfg.model.pretrained['image']
        state_dict_path = os.path.join(self.cfg.model.pretrained_path, state_dict_name)
        self.teacher = load_pretrained_d2v_model(state_dict_path=state_dict_path, keep_decoder=False)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        teacher_mode = self.teacher.cfg.supported_modality.name
        teacher_extra_tokens = self.teacher.modality_encoders[teacher_mode].modality_cfg.num_extra_tokens
        del self.teacher.modality_encoders # we share it between teacher and student
        
        self.model = KDMMData2Vec(cfg=self.cfg.model)
        
        assert self.model.modality_encoders['image'].modality_cfg.num_extra_tokens == teacher_extra_tokens, \
            f"Extra tokens mismatch: {self.model.modality_encoders['image'].modality_cfg.num_extra_tokens} != {teacher_extra_tokens} " \
                "between student and teacher model for modality 'image'" # TODO: add support for other modalities
        
        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict,
                          features_only=False,
                          return_encoder_output=True,
                          unmasked_feature_extractor_only=self.inverse_masked_kd)
        # "unmasked_feature_extractor_only" has higher precedence than "features_only" and "return_encoder_output"

    def training_step(self, batch:Dict[str, Any], batch_idx):
        if 'target' in batch:
            batch.pop('target') # unused, layer activations are the targets
        output_dict = self(batch) # call "forward"

        # in output_dict, because "return_encoder_output=True" in "forward"
        precomputed_encoder_output = output_dict['encoder_output']

        with torch.no_grad():
            target = self.teacher.extract_features(
                    source=batch['image'],
                    mode=None, # determined automatically in model
                    padding_mask=None, # the padding mask is provided in the precomputed_encoder_output and used by the teacher model
                    mask=False, # we are creating targets from a teacher model for the student model, so no mask
                    remove_extra_tokens=self.d2v_masking, # "decoder_input" in d2v (decoder in student model) removes extra tokens
                    # ... so we need to remove them from the teacher output as well if we mask the student input. If not, then we do not need to remove them
                    # because we also regress the extra tokens, if they are present.
                    precomputed_encoder_output=precomputed_encoder_output,
                    # inverse masked kd means we do masking for the student base on attention scores from the teacher
                    # ... so it is "masked_kd" from the teacher's perspective
                    masked_kd=self.inverse_masked_kd,
                    return_final_attn_scores=self.cfg.model.final_attn_layer_saliency_score, # ignored if inverse_masked_kd=False
                )
        
        if not self.inverse_masked_kd:
            target = target['layer_results']
            target = prepare_output(target, Modality.IMAGE)

        if self.d2v_masking:
            pred = output_dict['x']

            if self.cfg.model.clone_batch > 1:
                target = target.repeat_interleave(self.cfg.model.clone_batch, 0)

            masked_b = output_dict['mask'].mask.bool()
            assert pred.size(1) == masked_b.size(1), f"Size mismatch: {pred.size(1)} != {masked_b.size(1)}"
            assert target.size(1) == masked_b.size(1), f"Size mismatch: {target.size(1)} != {masked_b.size(1)}"
            pred = pred[masked_b]
            target = target[masked_b]
        elif self.masked_kd:
            pred = output_dict['layer_results'] # already prepared

        elif self.inverse_masked_kd:
            x_unmasked_tokens_only, keep_timesteps = get_max_saliency_patches(
                frac_keep_tokens=self.cfg.model.frac_keep_tokens,
                attn_results=target['attn_scores'],
                extractor_out=precomputed_encoder_output,
                feature_extractor=self.model.modality_encoders[batch['modes'][0].name.lower()],
            )
            
            precomputed_encoder_output["x"] = x_unmasked_tokens_only

            target = prepare_salient_patches( # "target" prepared now
                layer_results=target['layer_results'],
                keep_timesteps=keep_timesteps,
                mode=batch['modes'][0],
                norm_first=self.cfg.model.norm_first,
            )

            pred = self.model(
                modes=None, # ignored if "precomputed_encoder_output" provided, which is the case here
                features_only=False,
                return_encoder_output=False,
                precomputed_encoder_output=precomputed_encoder_output,)
            pred = pred['layer_results']
            pred = prepare_output(pred, Modality.IMAGE)
        else:
            pred = output_dict['layer_results']
            pred = prepare_output(pred, Modality.IMAGE)
        

        loss = self.kd_loss(input=pred, target=target)
        self.log("train/loss", loss, prog_bar=True)
        if batch['modes'][0] == Modality.IMAGE:
            self.log("train/loss_img", loss, prog_bar=True)
        elif batch['modes'][0] == Modality.AUDIO:
            self.log("train/loss_audio", loss, prog_bar=True)
        else:
            self.log("train/loss_text", loss, prog_bar=True)
        return loss
                
    
    def kd_loss(self, input:torch.Tensor, target:torch.Tensor) -> float:
        input = input.contiguous()
        input = input.view(-1, input.size(-1)).float() # (B, D, C) -> (B*D, C)
        target = target.contiguous()
        target = target.view(-1, target.size(-1)).float() # (B, D, C) -> (B*D, C)

        assert input.shape == target.shape # this must be the case

        return F.mse_loss(input, target, reduction="none").float().mean()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr,
                                      betas=tuple(self.cfg.optimizer.betas),
                                      eps=self.cfg.optimizer.eps,
                                      weight_decay=self.cfg.optimizer.weight_decay)
        if self.cfg.optimizer.warmup:
            name = self.cfg.optimizer_schedule.type
            if name == 'cosine':
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
                    num_training_steps=self.cfg.optimizer_schedule.max_steps,
                )
            else:
                scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
                )
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": name}]
        else:
            return optimizer


@dataclass
class PretrainedStateDictsConfig():
    audio:str = 'base_libri.pt'
    image:str = 'base_imagenet.pt'
    text:str = 'nlp_base.pt'

@dataclass
class BlockInitConfig():
    init_from: Optional[Modality] = None # if None, then no blocks are initialized
    init_type: Optional[str] = None # 'attention' or 'block', only relevant if "init_from" not None
    block_indices: Optional[List[int]] = None # if None, then all blocks are initialized
    freeze_blocks: Optional[List[int]] = None # if None, then all blocks are frozen, if empty list, then no blocks are frozen

@dataclass
class KDMMData2VecConfig():
    pretrained_path:str = '../models'
    pretrained: PretrainedStateDictsConfig = field(default_factory=PretrainedStateDictsConfig)

    supported_modalities: List[Modality] = field(default_factory=lambda: [Modality.AUDIO, Modality.IMAGE, Modality.TEXT])

    block_init_cfg: BlockInitConfig = field(default_factory=BlockInitConfig)

    mask: bool = False
    d2v_masking: bool = False

    # MaskedKD, relevant if "mask" is True and "d2v_masking" is False

    # if True, then the student gets the masked input, and the saliency score is computed from the teacher output
    inverse_masked_kd: bool = False
    frac_keep_tokens: float = 0.6
    # whether to normalize all timesteps (True), or only the ones that are kept (False)
    norm_first: bool = False
    # whether to use the final attention layer saliency score for masking (True) or the average of all layers (False)
    final_attn_layer_saliency_score: bool = False

    embed_dim: int = 768

    clone_batch: int = 1

    depth: int = 8
    num_heads: int = 12
    mlp_ratio: float = 4
    encoder_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    post_mlp_drop: float = 0.1
    norm_eps: float = 1e-6
    norm_affine: bool = True
    layer_norm_first: bool = False
    dropout_input: float = 0.0
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    layerdrop: float = 0.0

    seed: int = 42

class KDMMData2Vec(nn.Module):
    def __init__(self,
                 cfg: KDMMData2VecConfig,
                 ):
        super(KDMMData2Vec, self).__init__()
        self.cfg = cfg
        self.supported_modalities = cfg.supported_modalities
        self.fine_tuning = False

        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)

        if self.cfg.block_init_cfg.init_from is None:
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

        # init pretrained later, so that they are not part of the model's parameters when model is initialized
        self._init_from_pretrained()
        assert hasattr(self, 'blocks'), "Blocks must be initialized before initializing the model."
        
    def make_block(self, drop_path, dim=None, heads=None):
        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        return AltBlock(
            self.cfg.embed_dim if dim is None else dim,
            self.cfg.num_heads if heads is None else heads,
            self.cfg.mlp_ratio,
            qkv_bias=True,
            drop=self.cfg.encoder_dropout,
            attn_drop=self.cfg.attention_dropout,
            mlp_drop=self.cfg.activation_dropout,
            post_mlp_drop=self.cfg.post_mlp_drop,
            drop_path=drop_path,
            norm_layer=make_layer_norm,
            layer_norm_first=self.cfg.layer_norm_first,
            ffn_targets=True,
        )

    def _init_from_pretrained(self) -> None:
        modality_encoders = {}
        for mode in self.supported_modalities: # mode is instance of Modality
            state_dict_name = self.cfg.pretrained[mode.name.lower()]
            logger.info(f'Loading modality encoder for: {mode}')
            state_dict_path = os.path.join(self.cfg.pretrained_path, state_dict_name)
            # if we are masking the input to the student model, then we need to keep the decoder
            d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path, keep_decoder=self.cfg.mask and self.cfg.d2v_masking)
            mode_feature_extractor = d2v_model.modality_encoders[mode.name]
            modality_encoders[mode.name.lower()] = mode_feature_extractor
            
            for name, module in mode_feature_extractor.named_children():
                total_params = sum(p.numel() for p in module.parameters())
                logger.info(f"{name} has {total_params} parameters")

            if self.cfg.block_init_cfg.init_from is not None and mode == self.cfg.block_init_cfg.init_from:
                self._init_blocks(d2v_model=d2v_model)

        self.modality_encoders:nn.ModuleDict[str, nn.Module] = nn.ModuleDict(modality_encoders)
        self._freeze(self.modality_encoders)
        self.modality_encoders.eval()

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
        remove_extra_tokens:bool=False,
        precomputed_mask=None,
        return_encoder_output:bool=False,
        features_only:bool=False,
        unmasked_feature_extractor_only:bool=False,
        precomputed_encoder_output:Optional[Dict[str, torch.Tensor]]=None,
    ):
        mask_condition = self.cfg.mask and not features_only
        d2v_masking = mask_condition and self.cfg.d2v_masking
        masked_kd = mask_condition and not self.cfg.d2v_masking and not self.cfg.inverse_masked_kd

        if precomputed_encoder_output is None:
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

            feature_extractor:ModalitySpecificEncoder = self.modality_encoders[mode]
            with torch.no_grad() if not self.fine_tuning else contextlib.ExitStack():
                x_local = feature_extractor.local_features(source) # if we do d2v masking, we reuse the local features
                extractor_out_unmasked = feature_extractor.contextualized_features(
                    x=x_local,
                    padding_mask=padding_mask,
                    mask=False,
                    remove_masked=False,
                )
                if unmasked_feature_extractor_only: # is only the case for inverse masked KD
                    # we return a dict here, as we do not need to change the "training_step" function of the PL module that extensive
                    return {
                        "encoder_output": extractor_out_unmasked
                    }
                
                if d2v_masking:
                    extractor_out_masked = feature_extractor.contextualized_features(
                        x=x_local,
                        padding_mask=padding_mask,
                        mask=True,
                        remove_masked=True,
                        clone_batch=self.cfg.clone_batch,
                        mask_seeds=None,
                        precomputed_mask=precomputed_mask,
                    )
                    extractor_out = extractor_out_masked # if d2v masking, then student transformer layer use masked input
                else:
                    extractor_out = extractor_out_unmasked
        
        else:
            extractor_out = precomputed_encoder_output

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        if masked_kd:
            attn_results = []
        
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

                block_out = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                    return_att_scores=masked_kd,
                )
                if masked_kd:
                    x, lr, attn = block_out
                    if not self.cfg.final_attn_layer_saliency_score or i == len(self.blocks) - 1:
                        # if we only use the last attn layer scores, then we only append if we are at the last layer
                        attn_results.append(attn)
                else:
                    x, lr = block_out
                
                if not features_only:
                    layer_results.append(lr)

        if masked_kd:
            assert not d2v_masking, "MaskedKD and d2v masking are mutually exclusive."
            x_unmasked_tokens_only, keep_timesteps = get_max_saliency_patches(
                frac_keep_tokens=self.cfg.frac_keep_tokens,
                attn_results=attn_results,
                extractor_out=extractor_out, # in this case must be "extractor_out_unmasked" (d2v_masking=False)
                feature_extractor=feature_extractor,
            )
            
            extractor_out_unmasked['x'] = x_unmasked_tokens_only

            layer_results = prepare_salient_patches(
                layer_results=layer_results,
                keep_timesteps=keep_timesteps,
                mode=modes[0],
                norm_first=self.cfg.norm_first,
            )

        if self.norm is not None:
            x = self.norm(x)

        # we only use the decoder for d2v masking, else: we return here
        if not d2v_masking:
            if remove_extra_tokens:
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feature_extractor.modality_cfg.num_extra_tokens :
                    ]

            out = {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }
            if return_encoder_output:
                # teacher gets all tokens. Only if we do MaskedKD, then the teacher only gets the unmasked tokens
                out["encoder_output"] = extractor_out_unmasked
            return out
    
        assert hasattr(feature_extractor, 'decoder') and feature_extractor.decoder is not None, \
            "Decoder must be present in the feature extractor for masking the student input."
        # expands input back to original size -> adds masked time steps back
        # decoder weight are frozen as part of "self._freeze(self.modality_encoders)" in "_init_from_pretrained"
        # no @torch_no_grad() here, because we need to compute the loss,
        # ... and gradients need to flow through the decoder to the blocks!
        x = self.forward_decoder(
            x,
            feature_extractor,
            feature_extractor.decoder,
            encoder_mask,
        )

        out = {
            "x": x,
            "padding_mask": masked_padding_mask,
            "mask": encoder_mask,
        }
        if return_encoder_output:
            out["encoder_output"] = extractor_out_unmasked # for the teacher model, always unmasked

        return out


    def forward_decoder(
        self,
        x,
        feature_extractor,
        decoder,
        mask_info,
    ):
        x = feature_extractor.decoder_input(x, mask_info)
        x = decoder(*x)

        return x
    

    def extract_features(
        self, audio=None, image=None, text=None, modes:List[Modality]=None, padding_mask=None, remove_extra_tokens=True
    ):
        res = self.forward(
            audio=audio,
            image=image,
            text=text,
            modes=modes,
            padding_mask=padding_mask,
            remove_extra_tokens=remove_extra_tokens,
            features_only=True,
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

    def _init_blocks(self, d2v_model:Data2VecMultiModel) -> None:
        init_cfg:BlockInitConfig = self.cfg.block_init_cfg
        
        if init_cfg.block_indices is None:
            take_block_indices = [i for i in range(self.cfg.depth)]
        else:
            take_block_indices = init_cfg.block_indices

        logger.info(f"Initializing blocks from pretrained mode: {init_cfg.init_from}")
        logger.info(f"Init type: {init_cfg.init_type}")
        logger.info(f'Taking pretrained block indices: {take_block_indices}')

        if init_cfg.init_type == 'attention':
            dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)
            self.blocks = [self.make_block(dpr[i]) for i in range(self.cfg.depth)]
        else:
            self.blocks = []

        for idx in take_block_indices:
            if init_cfg.init_type == 'attention':
                self.blocks[idx].attn = d2v_model.blocks[idx].attn
            else:
                self.blocks.append(d2v_model.blocks[idx])

        n_blocks_missing = self.cfg.depth - len(take_block_indices)
        if n_blocks_missing > 0:
            assert init_cfg.init_type == 'block', "Only 'block' initialization supports adding new blocks"
            logger.info(f"Adding {n_blocks_missing} new blocks")
            dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, n_blocks_missing)
            for i in range(n_blocks_missing):
                self.blocks.append(self.make_block(dpr[i]))

        self.blocks = nn.ModuleList(self.blocks)

        if init_cfg.freeze_blocks is None:
            if init_cfg.init_type == 'attention':
                logger.info("Freezing all attention blocks")
                self.freeze_attention_blocks()
            else:
                logger.info("Freezing all blocks")
                self._freeze(self.blocks)
        elif len(init_cfg.freeze_blocks) == 0:
            pass # do not freeze any blocks
        else:
            if init_cfg.init_type == 'attention':
                logger.info(f"Freezing attention block indices: {init_cfg.freeze_blocks}")
                for idx in init_cfg.freeze_blocks:
                    self._freeze(self.blocks[idx].attn)
            else:
                logger.info(f"Freezing block indices: {init_cfg.freeze_blocks}")
                for idx in init_cfg.freeze_blocks:
                    self._freeze(self.blocks[idx])
        
    
    def freeze_attention_blocks(self):
        for block in self.blocks:
            self._freeze(block.attn)

    def unfreeze_attention_blocks(self):
        for block in self.blocks:
            self._unfreeze(block.attn)

    def _freeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def _unfreeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = True
        module.train()


    def prepare_fine_tuning(self, keep_modes:List[Modality]) -> None:
        self.cfg.clone_batch = 1
        self.fine_tuning = True
        self.cfg.mask = False
        self._remove_modalities_except(keep_modes=keep_modes)
        self._unfreeze(self.modality_encoders)

    def _remove_modalities_except(self, keep_modes:List[Modality]) -> None:
        """
        Removes all modalities from the model except the ones specified.
        Useful when fine-tuning the model on downstream task
        involving only a subset of the supported modalities.
        """
        # comparison done on name basis, as on "enum" basis yields problems after serialization
        keep_modes = [mode.name.lower() for mode in keep_modes]
        for modality in self.supported_modalities:
            modality_str = modality.name.lower()
            if modality_str not in keep_modes:
                del self.modality_encoders[modality_str] # includes removing the decoder
            else:
                if hasattr(self.modality_encoders[modality_str], 'decoder'):
                    del self.modality_encoders[modality_str].decoder # not needed in any case
