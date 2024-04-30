import torch.nn as nn
from typing import *
from examples.data2vec.data.modality import Modality
from examples.data2vec.models.data2vec_text_classification import Data2VecTextClassificationConfig
from examples.data2vec.models.mm_d2v import KDMMData2Vec, KDData2VecPreTrainingLightningModule
from fairseq.models.roberta.model import RobertaClassificationHead
from fairseq.models import BaseFairseqModel, register_model
import logging

logger = logging.getLogger(__name__)

@register_model(
    "mm_data2vec_text_classification", dataclass=Data2VecTextClassificationConfig
)
class TextClassificationModel(BaseFairseqModel):
    def __init__(self, cfg:Data2VecTextClassificationConfig):
        super().__init__()
        self.cfg = cfg

        self.model:KDMMData2Vec = KDData2VecPreTrainingLightningModule.load_from_checkpoint(self.cfg.model_path).model
        self.model.remove_modalities_except(keep_modes=[Modality.TEXT])

        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, cfg: Data2VecTextClassificationConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)
    
    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        embed_dim = 768
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=embed_dim,
            inner_dim=inner_dim or embed_dim,
            num_classes=num_classes,
            activation_fn=self.cfg.pooler_activation_fn,
            pooler_dropout=self.cfg.pooler_dropout,
            q_noise=self.cfg.quant_noise_pq,
            qn_block_size=self.cfg.quant_noise_pq_block_size,
            do_spectral_norm=self.cfg.spectral_norm_classification_head,
        )

    def forward(
        self,
        source,
        id,
        padding_mask,
        features_only=True,
        remove_extra_tokens=True,
        classification_head_name=None,
    ):
        encoder_out = self.model(
            text=source,
            mode=Modality.TEXT,
            padding_mask=padding_mask,
            mask=False,
            features_only=True,
            remove_extra_tokens=False,
        )
        logits = self.classification_heads[classification_head_name](encoder_out["x"])
        return logits, encoder_out
