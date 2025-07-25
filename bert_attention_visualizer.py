import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AttentionData:
    tokens: List[str]
    attention_weights: Tuple[torch.Tensor]
    hidden_states: Tuple[torch.Tensor]
    input_ids: torch.Tensor

class BERTAttentionVisualizer:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
    def _load_model(self):
        print(f"Loading {self.model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(
            self.model_name,
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation="eager"
        ).to(self.device).eval()
    def analyze_text(self, text: str) -> AttentionData:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return AttentionData(
            tokens=tokens,
            attention_weights=outputs.attentions,
            hidden_states=outputs.hidden_states,
            input_ids=inputs['input_ids']
        )
    def get_attention_matrix(self, data: AttentionData, layer: int, head: int, 
                           exclude_special: bool = True) -> Tuple[np.ndarray, List[str]]:
        attention = data.attention_weights[layer][0, head].cpu().numpy()
        tokens = data.tokens
        if exclude_special:
            mask = self._get_content_mask(data.input_ids[0])
            attention = attention[mask][:, mask]
            tokens = [t for t, m in zip(tokens, mask) if m]
        return attention, tokens
    def get_layer_attention_patterns(self, data: AttentionData, layer: int) -> Dict[str, np.ndarray]:
        patterns = {}
        for head in range(self.model.config.num_attention_heads):
            attention = data.attention_weights[layer][0, head].cpu().numpy()
            patterns[f'head_{head}'] = attention
        return patterns
    def _get_content_mask(self, input_ids: torch.Tensor) -> np.ndarray:
        special_tokens = {101, 102, 0}
        return ~np.isin(input_ids.cpu().numpy(), list(special_tokens))

class AttentionFormatter:
    @staticmethod
    def print_tokens(tokens: List[str], highlight_idx: Optional[int] = None):
        print("\n" + "="*60)
        print("TOKENIZATION")
        print("="*60)
        for i, token in enumerate(tokens):
            marker = "*" if token in ['[CLS]', '[SEP]', '[PAD]'] else " "
            if highlight_idx == i:
                print(f"\033[92m{i:3d} {marker} {token:<20} ← TARGET\033[0m")
            else:
                print(f"{i:3d} {marker} {token:<20}")
    @staticmethod
    def print_attention_matrix(attention: np.ndarray, tokens: List[str], layer: int, head: int):
        print(f"\n{'='*60}")
        print(f"ATTENTION: Layer {layer+1}/12, Head {head+1}/12")
        print(f"{'='*60}")
        print(f"{'':15}", end='')
        for token in tokens[:8]:
            print(f"{token[:7]:^8}", end='')
        if len(tokens) > 8:
            print("...", end='')
        print()
        for i, from_token in enumerate(tokens[:8]):
            print(f"{from_token[:12]:12}", end='')
            for j in range(min(8, len(tokens))):
                val = attention[i, j]
                if val > 0.15:
                    print(f"\033[1m{val:^8.3f}\033[0m", end='')
                else:
                    print(f"{val:^8.3f}", end='')
            if len(tokens) > 8:
                print("...", end='')
            print()
        if len(tokens) > 8:
            print("...")
        print(f"\nStats: max={attention.max():.3f}, mean={attention.mean():.3f}")

    @staticmethod
    def print_attention_patterns(data: AttentionData, token_idx: int):
        token = data.tokens[token_idx]
        print(f"\n{'='*60}")
        print(f"ATTENTION TO: '{token}' (position {token_idx})")
        print(f"{'='*60}")
        all_layer_attention = []
        for layer in range(12):
            attention = data.attention_weights[layer][0].cpu().numpy()
            avg_attention = attention[:, :, token_idx].mean(axis=0)
            all_layer_attention.append(avg_attention)
            print(f"\nLayer {layer+1}:")
            top_indices = np.argsort(avg_attention)[-3:][::-1]
            for idx in top_indices:
                score = avg_attention[idx]
                bar = '█' * int(score * 20)
                print(f"  {data.tokens[idx]:15} {score:.3f} {bar}")
        all_layer_attention = np.array(all_layer_attention)
        mean_attention = all_layer_attention.mean(axis=0)
        top_overall = np.argsort(mean_attention)[-3:][::-1]
        print(f"\n{'='*40}")
        print("TOP ATTENDING TOKENS (across all layers):")
        for idx in top_overall:
            score = mean_attention[idx]
            print(f"  {data.tokens[idx]:15} {score:.3f}")

    @staticmethod
    def print_attention_summary(data: AttentionData):
        print(f"\n{'='*60}")
        print("ATTENTION SUMMARY (all layers/heads)")
        print(f"{'='*60}")
        attention_stack = torch.stack(data.attention_weights).squeeze(1).cpu().numpy()
        n_layers, n_heads = attention_stack.shape[:2]
        max_per_head = attention_stack.max(axis=(-1, -2))
        mean_per_head = attention_stack.mean(axis=(-1, -2))
        print("\nMax attention per layer/head:")
        print("     ", end='')
        for h in range(min(n_heads, 8)):
            print(f" H{h+1:<2d}", end='')
        print("  ..." if n_heads > 8 else "")
        for l in range(n_layers):
            print(f"L{l+1:<2d}: ", end='')
            for h in range(min(n_heads, 8)):
                val = max_per_head[l, h]
                if val > 0.5:
                    print(f"\033[1m{val:5.2f}\033[0m", end='')
                else:
                    print(f"{val:5.2f}", end='')
            print("  ..." if n_heads > 8 else "")
        flat_idx = np.argmax(max_per_head)
        max_layer, max_head = np.unravel_index(flat_idx, max_per_head.shape)
        print(f"\nMost focused: Layer {max_layer+1}, Head {max_head+1} "
              f"(max={max_per_head[max_layer, max_head]:.3f}")
        entropy = -np.sum(attention_stack * np.log(attention_stack + 1e-9), axis=(-1, -2))
        max_entropy_idx = np.argmax(entropy)
        ent_layer, ent_head = np.unravel_index(max_entropy_idx, entropy.shape)
        print(f"Most distributed: Layer {ent_layer+1}, Head {ent_head+1} "
              f"(entropy={entropy[ent_layer, ent_head]:.3f}")
        
    @staticmethod
    def print_head_comparison(data: AttentionData, layer: int, heads: List[int] = None):
        if heads is None:
            heads = [0, 1, 2]
        print(f"\n{'='*60}")
        print(f"HEAD COMPARISON: Layer {layer+1}, Heads {heads}")
        print(f"{'='*60}")
        n_tokens = min(8, len(data.tokens))
        for i in range(n_tokens):
            print(f"\nFrom '{data.tokens[i]}':")
            for j in range(n_tokens):
                if i == j:
                    continue
                scores = []
                for h in heads:
                    attention = data.attention_weights[layer][0, h, i, j].item()
                    scores.append((h, attention))
                scores.sort(key=lambda x: x[1], reverse=True)
                if scores[0][1] > 0.1:
                    print(f"  → '{data.tokens[j]:12}' : ", end='')
                    for h, score in scores[:2]:
                        print(f"H{h+1}={score:.2f} ", end='')
                    print()

class HeadAnalyzer:
    def __init__(self, model: BertModel):
        self.model = model
        self.hidden_dim = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads

    def get_qkv_matrices(self, data: AttentionData, layer: int, head: int
                        ) -> Dict[str, np.ndarray]:
        hidden_state = data.hidden_states[layer]
        attn_layer = self.model.encoder.layer[layer].attention.self
        Q = attn_layer.query(hidden_state)
        K = attn_layer.key(hidden_state)
        V = attn_layer.value(hidden_state)
        batch_size, seq_len = hidden_state.shape[:2]
        for matrix, name in [(Q, 'Q'), (K, 'K'), (V, 'V')]:
            matrix = matrix.view(batch_size, seq_len, self.num_heads, self.head_dim)
            matrix = matrix.transpose(1, 2)
            if name == 'Q':
                Q = matrix
            elif name == 'K':
                K = matrix
            else:
                V = matrix
        return {
            'Q': Q[0, head].detach().cpu().numpy(),
            'K': K[0, head].detach().cpu().numpy(),
            'V': V[0, head].detach().cpu().numpy()
        }
    
    def print_head_analysis(self, data: AttentionData, layer: int, head: int):
        qkv = self.get_qkv_matrices(data, layer, head)
        print(f"\n{'='*60}")
        print(f"HEAD INTERNALS: Layer {layer+1}, Head {head+1}")
        print(f"{'='*60}")
        print(f"Dimension: {self.head_dim} per head")
        for matrix_name, matrix in qkv.items():
            print(f"\n{matrix_name} Matrix (first 5 dims):")
            print("-" * 40)
            for i, token in enumerate(data.tokens[:5]):
                values = matrix[i, :5]
                norm = np.linalg.norm(matrix[i])
                print(f"{token:12} [{values[0]:5.2f} {values[1]:5.2f} {values[2]:5.2f} "
                      f"{values[3]:5.2f} {values[4]:5.2f}...] ||{matrix_name}||={norm:.2f}")
        scores = np.matmul(qkv['Q'], qkv['K'].T) / np.sqrt(self.head_dim)
        attention_weights = data.attention_weights[layer][0, head].cpu().numpy()
        print(f"\nPre-softmax scores (Q·K^T/√d):")
        print("-" * 40)
        for i in range(min(5, len(data.tokens))):
            print(f"{data.tokens[i]:12}", end='')
            for j in range(min(5, len(data.tokens))):
                print(f"{scores[i,j]:6.2f}", end='')
            print("...")
        print(f"\nPost-softmax attention (from model):")
        print("-" * 40)
        for i in range(min(5, len(data.tokens))):
            print(f"{data.tokens[i]:12}", end='')
            for j in range(min(5, len(data.tokens))):
                val = attention_weights[i,j]
                if val > 0.1:
                    print(f"\033[1m{val:6.3f}\033[0m", end='')
                else:
                    print(f"{val:6.3f}", end='')
            print("...")
        print(f"\nVector Similarities:")
        print("-" * 40)
        for name, matrix in qkv.items():
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            normalized = matrix / (norms + 1e-8)
            sim_matrix = np.matmul(normalized, normalized.T)
            np.fill_diagonal(sim_matrix, -1)
            max_sim_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            max_sim = sim_matrix[max_sim_idx]
            print(f"{name}: Most similar pair = "
                  f"'{data.tokens[max_sim_idx[0]]}' & '{data.tokens[max_sim_idx[1]]}' "
                  f"(cos={max_sim:.3f})")
        print(f"\nDimension Statistics:")
        print("-" * 40)
        for name, matrix in qkv.items():
            var_per_dim = np.var(matrix, axis=0)
            top_var_dims = np.argsort(var_per_dim)[-3:][::-1]
            print(f"{name}: Most variable dims = {top_var_dims[0]}, {top_var_dims[1]}, {top_var_dims[2]} "
                  f"(var={var_per_dim[top_var_dims[0]]:.3f}, "
                  f"{var_per_dim[top_var_dims[1]]:.3f}, "
                  f"{var_per_dim[top_var_dims[2]]:.3f})")

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('text', help='Text to analyze')
    parser.add_argument('-l', '--layer', type=int, default=0,
                        help='Layer index (0-11)')
    parser.add_argument('-H', '--head', type=int, default=0,
                        help='Head index (0-11)')
    parser.add_argument('--tokens', action='store_true',
                        help='Show only tokenization')
    parser.add_argument('--matrix', action='store_true',
                        help='Show attention matrix')
    parser.add_argument('--focus', type=int, metavar='TOKEN_IDX',
                        help='Show attention to specific token')
    parser.add_argument('--internals', action='store_true',
                        help='Show head internals (Q/K/V)')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary across all layers/heads')
    parser.add_argument('--compare-heads', nargs='*', type=int, metavar='HEAD',
                        help='Compare multiple heads in a layer')
    parser.add_argument('--include-special', action='store_true',
                        help='Include special tokens ([CLS], [SEP])')
    parser.add_argument('--model', default='bert-base-uncased',
                        help='Model name (default: bert-base-uncased)')
    args = parser.parse_args()
    visualizer = BERTAttentionVisualizer(args.model)
    formatter = AttentionFormatter()
    data = visualizer.analyze_text(args.text)
    if args.tokens:
        formatter.print_tokens(data.tokens)
    elif args.focus is not None:
        if 0 <= args.focus < len(data.tokens):
            formatter.print_tokens(data.tokens, highlight_idx=args.focus)
            formatter.print_attention_patterns(data, args.focus)
        else:
            print(f"Error: Token index {args.focus} out of range (0-{len(data.tokens)-1})")
    elif args.internals:
        analyzer = HeadAnalyzer(visualizer.model)
        analyzer.print_head_analysis(data, args.layer, args.head)
    elif args.summary:
        formatter.print_attention_summary(data)
    elif args.compare_heads is not None:
        heads = args.compare_heads if args.compare_heads else None
        formatter.print_head_comparison(data, args.layer, heads)
    else:
        attention, tokens = visualizer.get_attention_matrix(
            data, args.layer, args.head, not args.include_special
        )
        formatter.print_attention_matrix(attention, tokens, args.layer, args.head)

if __name__ == "__main__":
    main()