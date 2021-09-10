from transformers import (
    BertModel,
    BertForSequenceClassification,
)
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead


class BertForSimCSE(BertPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
