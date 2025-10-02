
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

class BERTModel:

    def __init__(self, model_name='bert-base-uncased', device='auto'):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.tokenizer = None
        self.model = None
        self._load()

    def _resolve_device(self, requested):
        if requested == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but not available')
            return 'cuda'
        if requested == 'cpu':
            return 'cpu'
        raise ValueError(f"Unsupported device: '{requested}'")

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation='eager'
        ).to(self.device).eval()

    def forward(self, text, return_dict=True):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        if not return_dict:
            return outputs

        return {
            'tokens': tokens,
            'input_ids': inputs['input_ids'],
            'attention': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'num_layers': self.model.config.num_hidden_layers,
            'num_heads': self.model.config.num_attention_heads,
            'text': text
        }

    def get_config(self):
        return {
            'model_name': self.model_name,
            'num_layers': self.model.config.num_hidden_layers,
            'num_heads': self.model.config.num_attention_heads,
            'hidden_size': self.model.config.hidden_size,
            'head_dim': self.model.config.hidden_size // self.model.config.num_attention_heads,
            'device': self.device
        }
