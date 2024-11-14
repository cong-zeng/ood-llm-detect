import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name,output_hidden_states=False):
        super(TextEmbeddingModel, self).__init__()
        self.model_name = model_name
        if output_hidden_states:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def pooling(self, model_output, attention_mask, use_pooling='average',hidden_states=False):
        if hidden_states:
            model_output.masked_fill(~attention_mask[None,..., None].bool(), 0.0)
            if use_pooling == "average":
                emb = model_output.sum(dim=2) / attention_mask.sum(dim=1)[..., None]
            else:
                emb = model_output[:,:, 0]
            emb = emb.permute(1, 0, 2)
        else:
            model_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
            if use_pooling == "average":
                emb = model_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            elif use_pooling == "cls":
                emb = model_output[:, 0]
        return emb
    
    def forward(self, encoded_batch, use_pooling='average',hidden_states=False):
        if "t5" in self.model_name.lower():
            input_ids = encoded_batch['input_ids']
            decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
            model_output = self.model(**encoded_batch, 
                                  decoder_input_ids=decoder_input_ids)
        else:
            model_output = self.model(**encoded_batch)
        
        if 'bge' in self.model_name.lower() or 'mxbai' in self.model_name.lower():
            use_pooling = 'cls'
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        if isinstance(model_output, dict):
            if hidden_states:
                model_output = model_output["hidden_states"]
                model_output = torch.stack(model_output, dim=0)
            else:
                model_output = model_output["last_hidden_state"]
        
        emb = self.pooling(model_output, encoded_batch['attention_mask'], use_pooling,hidden_states)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    