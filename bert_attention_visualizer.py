import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import argparse
import json
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AttentionData:
    tokens: List[str]
    attention_weights: Tuple[torch.Tensor]
    hidden_states: Tuple[torch.Tensor]
    input_ids: torch.Tensor
    text: str = ""  # Store original text for reference

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
            input_ids=inputs['input_ids'],
            text=text
        )
    def get_attention_matrix(self, data: AttentionData, layer: int, head: int, exclude_special: bool = True) -> Tuple[np.ndarray, List[str]]:
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
    def print_detailed_attention_weights(data: AttentionData, layer: Optional[int] = None, 
                                        head: Optional[int] = None) -> Dict[str, Any]:
        """Print all attention weights with detailed position information"""
        output = {}
        layers = [layer] if layer is not None else range(len(data.attention_weights))
        
        for l in layers:
            output[f'layer_{l}'] = {}
            heads = [head] if head is not None else range(data.attention_weights[l].size(1))
            
            for h in heads:
                attention = data.attention_weights[l][0, h].cpu().numpy()
                output[f'layer_{l}'][f'head_{h}'] = {
                    'matrix': attention.tolist(),
                    'tokens': data.tokens,
                    'shape': attention.shape
                }
                
                print(f"\n{'='*80}")
                print(f"LAYER {l+1}, HEAD {h+1}")
                print(f"{'='*80}")
                print(f"Shape: {attention.shape}")
                print(f"Min: {attention.min():.6f}, Max: {attention.max():.6f}, Mean: {attention.mean():.6f}")
                
                # Print detailed matrix with positions
                print(f"\nPosition-wise attention weights:")
                header_label = "From\\To"
                print(f"{header_label:<8}", end="")
                for j, to_token in enumerate(data.tokens):
                    print(f"{j:6d}", end="")
                print()
                
                for i, from_token in enumerate(data.tokens):
                    print(f"{i:3d} {from_token[:4]:4}", end="")
                    for j in range(len(data.tokens)):
                        val = attention[i, j]
                        print(f"{val:6.3f}", end="")
                    print()
                    
                # Sparsity analysis
                zero_weights = np.sum(attention < 1e-6)
                total_weights = attention.size
                sparsity = zero_weights / total_weights
                print(f"\nSparsity analysis:")
                print(f"  Zero weights: {zero_weights}/{total_weights} ({sparsity*100:.2f}%)")
                
                # Top attention weights
                flat_indices = np.argsort(attention.flatten())[-10:][::-1]
                top_weights = [(divmod(idx, attention.shape[1]), attention.flatten()[idx]) 
                              for idx in flat_indices]
                
                print(f"\nTop 10 attention weights:")
                for (i, j), weight in top_weights:
                    print(f"  {i:2d} → {j:2d} : {weight:.6f} ({data.tokens[i]} → {data.tokens[j]})")
        
        return output

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
    
    def print_detailed_head_analysis(self, data: AttentionData, layer: int, head: int):
        """Enhanced head analysis with detailed value enumeration"""
        qkv = self.get_qkv_matrices(data, layer, head)
        print(f"\n{'='*80}")
        print(f"DETAILED HEAD ANALYSIS: Layer {layer+1}, Head {head+1}")
        print(f"{'='*80}")
        print(f"Dimension: {self.head_dim} per head")
        
        # Detailed QKV matrices with position indices
        for matrix_name, matrix in qkv.items():
            print(f"\n{matrix_name} Matrix (all dimensions):")
            print("-" * 60)
            print(f"{'Token':<15} {'Vector Values (first 10)':<50} ||{matrix_name}||")
            for i, token in enumerate(data.tokens):
                values = matrix[i]
                norm = np.linalg.norm(values)
                values_str = " ".join([f"{v:6.3f}" for v in values[:10]])
                print(f"{token:<15} {values_str:<50} {norm:.3f}")
                
        # Pre-softmax scores with detailed position information
        scores = np.matmul(qkv['Q'], qkv['K'].T) / np.sqrt(self.head_dim)
        attention_weights = data.attention_weights[layer][0, head].cpu().numpy()
        
        print(f"\nPre-softmax scores (Q·K^T/√d):")
        print("-" * 60)
        header_label = "From\\To"
        print(f"{header_label:<15}", end="")
        for j, to_token in enumerate(data.tokens):
            print(f"{j:8d}", end="")
        print()
        
        for i, from_token in enumerate(data.tokens):
            print(f"{i:3d} {from_token:<11}", end='')
            for j in range(len(data.tokens)):
                print(f"{scores[i,j]:8.3f}", end='')
            print()
            
        # Post-softmax attention with detailed position information
        print(f"\nPost-softmax attention (from model):")
        print("-" * 60)
        print(f"{header_label:<15}", end="")
        for j, to_token in enumerate(data.tokens):
            print(f"{j:8d}", end="")
        print()
        
        for i, from_token in enumerate(data.tokens):
            print(f"{i:3d} {from_token:<11}", end='')
            for j in range(len(data.tokens)):
                val = attention_weights[i,j]
                if val > 0.1:
                    print(f"\033[1m{val:8.3f}\033[0m", end='')
                else:
                    print(f"{val:8.3f}", end='')
            print()
            
        # Vector similarities with detailed information
        print(f"\nVector Similarities:")
        print("-" * 60)
        for name, matrix in qkv.items():
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            normalized = matrix / (norms + 1e-8)
            sim_matrix = np.matmul(normalized, normalized.T)
            np.fill_diagonal(sim_matrix, -1)
            
            print(f"\n{name} Vector Cosine Similarities:")
            print(f"{'Token1':<15} {'Token2':<15} {'Cosine Similarity'}")
            # Get top 10 similar pairs
            flat_indices = np.argsort(sim_matrix.flatten())[-11:][::-1]  # -11 to exclude diagonal
            similar_pairs = [(divmod(idx, sim_matrix.shape[1]), sim_matrix.flatten()[idx]) 
                            for idx in flat_indices]
            
            for (i, j), similarity in similar_pairs:
                if i != j:  # Skip diagonal
                    print(f"{data.tokens[i]:<15} {data.tokens[j]:<15} {similarity:.6f}")
                    
        # Dimension statistics with detailed information
        print(f"\nDimension Statistics:")
        print("-" * 60)
        for name, matrix in qkv.items():
            mean_per_dim = np.mean(matrix, axis=0)
            var_per_dim = np.var(matrix, axis=0)
            std_per_dim = np.std(matrix, axis=0)
            
            # Top variance dimensions
            top_var_dims = np.argsort(var_per_dim)[-5:][::-1]
            print(f"\n{name} Matrix Dimension Analysis:")
            print(f"{'Dim':<5} {'Mean':<10} {'Variance':<10} {'Std Dev':<10}")
            for dim in top_var_dims:
                print(f"{dim:<5} {mean_per_dim[dim]:<10.6f} {var_per_dim[dim]:<10.6f} {std_per_dim[dim]:<10.6f}")
                
        # Attention sparsity analysis
        flat_attention = attention_weights.flatten()
        zero_weights = np.sum(flat_attention < 1e-6)
        total_weights = flat_attention.size
        sparsity = zero_weights / total_weights
        
        print(f"\nAttention Sparsity Analysis:")
        print("-" * 60)
        print(f"Zero weights: {zero_weights}/{total_weights} ({sparsity*100:.2f}%)")
        print(f"Non-zero weights: {total_weights - zero_weights}")
        print(f"Min attention weight: {flat_attention.min():.6f}")
        print(f"Max attention weight: {flat_attention.max():.6f}")
        print(f"Mean attention weight: {flat_attention.mean():.6f}")
        print(f"Median attention weight: {np.median(flat_attention):.6f}")
        
        # Value enumeration for all attention weights
        print(f"\nValue Enumeration (sorted by magnitude):")
        print("-" * 60)
        sorted_indices = np.argsort(flat_attention)[::-1]  # Descending order
        print(f"{'Rank':<6} {'From':<6} {'To':<6} {'Weight':<10} {'Tokens'}")
        for i, idx in enumerate(sorted_indices[:20]):  # Top 20
            from_idx, to_idx = divmod(idx, attention_weights.shape[1])
            weight = flat_attention[idx]
            from_token = data.tokens[from_idx]
            to_token = data.tokens[to_idx]
            print(f"{i+1:<6} {from_idx:<6} {to_idx:<6} {weight:<10.6f} {from_token} → {to_token}")

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
        header_label = "From\To"
        print(f"{header_label:<12}", end='')
        for j in range(min(5, len(data.tokens))):
            print(f"{j:6d}", end='')
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

class DataExporter:
    """Export attention data in various formats for compression and pruning analysis"""
    
    @staticmethod
    def export_to_json(data: AttentionData, filepath: str):
        """Export all attention data to JSON format"""
        export_data = {
            'text': data.text,
            'tokens': data.tokens,
            'attention_weights': [],
            'hidden_states': []
        }
        
        # Export attention weights for all layers and heads
        for layer_idx, layer_weights in enumerate(data.attention_weights):
            layer_data = {
                'layer': layer_idx,
                'heads': []
            }
            for head_idx in range(layer_weights.size(1)):  # Number of heads
                head_data = {
                    'head': head_idx,
                    'attention_matrix': layer_weights[0, head_idx].cpu().numpy().tolist()
                }
                layer_data['heads'].append(head_data)
            export_data['attention_weights'].append(layer_data)
            
        # Export hidden states
        for layer_idx, hidden_state in enumerate(data.hidden_states):
            export_data['hidden_states'].append({
                'layer': layer_idx,
                'hidden_state': hidden_state[0].cpu().numpy().tolist()  # First batch item
            })
            
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Exported attention data to {filepath}")
        
    @staticmethod
    def export_attention_to_csv(data: AttentionData, filepath: str, 
                               layer: Optional[int] = None, head: Optional[int] = None):
        """Export attention weights to CSV format"""
        layers = [layer] if layer is not None else range(len(data.attention_weights))
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['layer', 'head', 'from_token_idx', 'to_token_idx', 'from_token', 'to_token', 'attention_weight']
            writer.writerow(header)
            
            # Write data
            for l in layers:
                heads = [head] if head is not None else range(data.attention_weights[l].size(1))
                for h in heads:
                    attention = data.attention_weights[l][0, h].cpu().numpy()
                    for i, from_token in enumerate(data.tokens):
                        for j, to_token in enumerate(data.tokens):
                            writer.writerow([l, h, i, j, from_token, to_token, attention[i, j]])
                            
        print(f"Exported attention weights to {filepath}")
        
    @staticmethod
    def export_sparsity_analysis(data: AttentionData, filepath: str):
        """Export sparsity analysis for all attention heads"""
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['layer', 'head', 'total_weights', 'zero_weights', 'sparsity_ratio'])
            
            for layer_idx, layer_weights in enumerate(data.attention_weights):
                for head_idx in range(layer_weights.size(1)):
                    attention = layer_weights[0, head_idx].cpu().numpy()
                    flat_attention = attention.flatten()
                    total_weights = flat_attention.size
                    zero_weights = np.sum(flat_attention < 1e-6)
                    sparsity = zero_weights / total_weights
                    
                    writer.writerow([layer_idx, head_idx, total_weights, zero_weights, sparsity])
                    
        print(f"Exported sparsity analysis to {filepath}")


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
    parser.add_argument('--detailed-internals', action='store_true',
                        help='Show detailed head internals with value enumeration')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary across all layers/heads')
    parser.add_argument('--compare-heads', nargs='*', type=int, metavar='HEAD',
                        help='Compare multiple heads in a layer')
    parser.add_argument('--include-special', action='store_true',
                        help='Include special tokens ([CLS], [SEP])')
    parser.add_argument('--model', default='bert-base-uncased',
                        help='Model name (default: bert-base-uncased)')
    parser.add_argument('--export-json', metavar='FILE',
                        help='Export all attention data to JSON file')
    parser.add_argument('--export-csv', metavar='FILE',
                        help='Export attention weights to CSV file')
    parser.add_argument('--export-sparsity', metavar='FILE',
                        help='Export sparsity analysis to CSV file')
    parser.add_argument('--detailed-weights', action='store_true',
                        help='Show detailed attention weights with position indices')
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
    elif args.detailed_weights:
        formatter.print_detailed_attention_weights(data, args.layer, args.head)
    elif args.internals:
        analyzer = HeadAnalyzer(visualizer.model)
        analyzer.print_head_analysis(data, args.layer, args.head)
    elif args.detailed_internals:
        analyzer = HeadAnalyzer(visualizer.model)
        analyzer.print_detailed_head_analysis(data, args.layer, args.head)
    elif args.summary:
        formatter.print_attention_summary(data)
    elif args.compare_heads is not None:
        heads = args.compare_heads if args.compare_heads else None
        formatter.print_head_comparison(data, args.layer, heads)
    elif args.export_json:
        exporter = DataExporter()
        exporter.export_to_json(data, args.export_json)
    elif args.export_csv:
        exporter = DataExporter()
        exporter.export_attention_to_csv(data, args.export_csv, args.layer, args.head)
    elif args.export_sparsity:
        exporter = DataExporter()
        exporter.export_sparsity_analysis(data, args.export_sparsity)
    else:
        attention, tokens = visualizer.get_attention_matrix(
            data, args.layer, args.head, not args.include_special
        )
        formatter.print_attention_matrix(attention, tokens, args.layer, args.head)


if __name__ == "__main__":
    main()