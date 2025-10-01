import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import argparse
import json
import csv
import warnings
warnings.filterwarnings('ignore')

class AttentionData:

    def __init__(self, tokens, attention_weights, hidden_states, input_ids, num_layers, num_heads, text=''):
        self.tokens = tokens
        self.attention_weights = attention_weights
        self.hidden_states = hidden_states
        self.input_ids = input_ids
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.text = text

class BERTAttentionVisualizer:

    def __init__(self, model_name='bert-base-uncased', device='auto'):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._load_model()

    def _resolve_device(self, requested):
        if requested == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but not available on this system')
            return 'cuda'
        if requested == 'cpu':
            return 'cpu'
        raise ValueError(f"Unsupported device option '{requested}'")

    def _load_model(self):
        print(f'Loading {self.model_name}...')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name, output_attentions=True, output_hidden_states=True, attn_implementation='eager').to(self.device).eval()

    def analyze_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for (k, v) in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        return AttentionData(tokens=tokens, attention_weights=outputs.attentions, hidden_states=outputs.hidden_states, input_ids=inputs['input_ids'], num_layers=num_layers, num_heads=num_heads, text=text)

    def get_attention_matrix(self, data, layer, head, exclude_special=True):
        if not 0 <= layer < data.num_layers:
            raise IndexError(f'Layer index {layer} out of range (0-{data.num_layers - 1})')
        if not 0 <= head < data.num_heads:
            raise IndexError(f'Head index {head} out of range (0-{data.num_heads - 1})')
        attention = data.attention_weights[layer][0, head].cpu().numpy()
        tokens = data.tokens
        if exclude_special:
            mask = self._get_content_mask(data.input_ids[0])
            attention = attention[mask][:, mask]
            tokens = [t for (t, m) in zip(tokens, mask) if m]
        return (attention, tokens)

    def get_layer_attention_patterns(self, data, layer):
        if not 0 <= layer < data.num_layers:
            raise IndexError(f'Layer index {layer} out of range (0-{data.num_layers - 1})')
        patterns = {}
        for head in range(data.num_heads):
            attention = data.attention_weights[layer][0, head].cpu().numpy()
            patterns[f'head_{head}'] = attention
        return patterns

    def _get_content_mask(self, input_ids):
        special_tokens = {101, 102, 0}
        return ~np.isin(input_ids.cpu().numpy(), list(special_tokens))

class AttentionFormatter:
    MAX_DISPLAY_TOKENS = 32

    @staticmethod
    def print_tokens(tokens, highlight_idx=None):
        print('\n' + '=' * 60)
        print('TOKENIZATION')
        print('=' * 60)
        max_tokens = AttentionFormatter.MAX_DISPLAY_TOKENS
        truncated = len(tokens) > max_tokens
        display_tokens = tokens[:max_tokens]
        for (i, token) in enumerate(display_tokens):
            marker = '*' if token in ['[CLS]', '[SEP]', '[PAD]'] else ' '
            if highlight_idx == i:
                print(f'\x1b[92m{i:3d} {marker} {token:<20} ← TARGET\x1b[0m')
            else:
                print(f'{i:3d} {marker} {token:<20}')
        if truncated:
            print(f'... truncated to first {max_tokens} tokens (of {len(tokens)}) ...')
            if highlight_idx is not None and highlight_idx >= max_tokens:
                print(f'(target index {highlight_idx} not shown)')

    @staticmethod
    def print_detailed_attention_weights(data, layer=None, head=None):
        output = {}
        layers = [layer] if layer is not None else range(data.num_layers)
        for l in layers:
            output[f'layer_{l}'] = {}
            heads = [head] if head is not None else range(data.num_heads)
            for h in heads:
                attention = data.attention_weights[l][0, h].cpu().numpy()
                output[f'layer_{l}'][f'head_{h}'] = {'matrix': attention.tolist(), 'tokens': data.tokens, 'shape': attention.shape}
                print(f"\n{'=' * 80}")
                print(f'LAYER {l + 1}, HEAD {h + 1}')
                print(f"{'=' * 80}")
                print(f'Shape: {attention.shape}')
                print(f'Min: {attention.min():.6f}, Max: {attention.max():.6f}, Mean: {attention.mean():.6f}')
                print(f'\nPosition-wise attention weights:')
                header_label = 'From\\To'
                token_indices = list(range(len(data.tokens)))
                if len(token_indices) > AttentionFormatter.MAX_DISPLAY_TOKENS:
                    token_indices = token_indices[:AttentionFormatter.MAX_DISPLAY_TOKENS]
                    print(f'(displaying first {AttentionFormatter.MAX_DISPLAY_TOKENS} tokens of {len(data.tokens)})')
                attention_subset = attention[np.ix_(token_indices, token_indices)]
                print(f'{header_label:<8}', end='')
                for idx in token_indices:
                    print(f'{idx:6d}', end='')
                print()
                for (row, token_idx) in enumerate(token_indices):
                    from_token = data.tokens[token_idx]
                    print(f'{token_idx:3d} {from_token[:4]:4}', end='')
                    for (col, _) in enumerate(token_indices):
                        val = attention_subset[row, col]
                        print(f'{val:6.3f}', end='')
                    print()
                zero_weights = np.sum(attention < 1e-06)
                total_weights = attention.size
                sparsity = zero_weights / total_weights
                print(f'\nSparsity analysis:')
                print(f'  Zero weights: {zero_weights}/{total_weights} ({sparsity * 100:.2f}%)')
                flat_indices = np.argsort(attention.flatten())[-10:][::-1]
                top_weights = [(divmod(idx, attention.shape[1]), attention.flatten()[idx]) for idx in flat_indices]
                print(f'\nTop 10 attention weights:')
                for ((i, j), weight) in top_weights:
                    print(f'  {i:2d} → {j:2d} : {weight:.6f} ({data.tokens[i]} → {data.tokens[j]})')
        return output

    @staticmethod
    def print_attention_matrix(attention, tokens, layer, head, total_layers, total_heads):
        print(f"\n{'=' * 60}")
        print(f'ATTENTION: Layer {layer + 1}/{total_layers}, Head {head + 1}/{total_heads}')
        print(f"{'=' * 60}")
        print(f"{'':15}", end='')
        for token in tokens[:8]:
            print(f'{token[:7]:^8}', end='')
        if len(tokens) > 8:
            print('...', end='')
        print()
        for (i, from_token) in enumerate(tokens[:8]):
            print(f'{from_token[:12]:12}', end='')
            for j in range(min(8, len(tokens))):
                val = attention[i, j]
                if val > 0.15:
                    print(f'\x1b[1m{val:^8.3f}\x1b[0m', end='')
                else:
                    print(f'{val:^8.3f}', end='')
            if len(tokens) > 8:
                print('...', end='')
            print()
        if len(tokens) > 8:
            print('...')
        print(f'\nStats: max={attention.max():.3f}, mean={attention.mean():.3f}')

    @staticmethod
    def print_attention_patterns(data, token_idx):
        token = data.tokens[token_idx]
        print(f"\n{'=' * 60}")
        print(f"ATTENTION TO: '{token}' (position {token_idx})")
        print(f"{'=' * 60}")
        all_layer_attention = []
        for layer in range(data.num_layers):
            attention = data.attention_weights[layer][0].cpu().numpy()
            avg_attention = attention[:, :, token_idx].mean(axis=0)
            all_layer_attention.append(avg_attention)
            print(f'\nLayer {layer + 1}:')
            top_indices = np.argsort(avg_attention)[-3:][::-1]
            for idx in top_indices:
                score = avg_attention[idx]
                bar = '█' * int(score * 20)
                print(f'  {data.tokens[idx]:15} {score:.3f} {bar}')
        all_layer_attention = np.array(all_layer_attention)
        mean_attention = all_layer_attention.mean(axis=0)
        top_overall = np.argsort(mean_attention)[-3:][::-1]
        print(f"\n{'=' * 40}")
        print('TOP ATTENDING TOKENS (across all layers):')
        for idx in top_overall:
            score = mean_attention[idx]
            print(f'  {data.tokens[idx]:15} {score:.3f}')

    @staticmethod
    def print_attention_summary(data):
        print(f"\n{'=' * 60}")
        print('ATTENTION SUMMARY (all layers/heads)')
        print(f"{'=' * 60}")
        attention_stack = torch.stack(data.attention_weights).squeeze(1).cpu().numpy()
        (n_layers, n_heads) = attention_stack.shape[:2]
        max_per_head = attention_stack.max(axis=(-1, -2))
        mean_per_head = attention_stack.mean(axis=(-1, -2))
        print('\nMax attention per layer/head:')
        print('     ', end='')
        for h in range(min(n_heads, 8)):
            print(f' H{h + 1:<2d}', end='')
        print('  ...' if n_heads > 8 else '')
        for l in range(n_layers):
            print(f'L{l + 1:<2d}: ', end='')
            for h in range(min(n_heads, 8)):
                val = max_per_head[l, h]
                if val > 0.5:
                    print(f'\x1b[1m{val:5.2f}\x1b[0m', end='')
                else:
                    print(f'{val:5.2f}', end='')
            print('  ...' if n_heads > 8 else '')
        flat_idx = np.argmax(max_per_head)
        (max_layer, max_head) = np.unravel_index(flat_idx, max_per_head.shape)
        print(f'\nMost focused: Layer {max_layer + 1}, Head {max_head + 1} (max={max_per_head[max_layer, max_head]:.3f}')
        entropy = -np.sum(attention_stack * np.log(attention_stack + 1e-09), axis=(-1, -2))
        max_entropy_idx = np.argmax(entropy)
        (ent_layer, ent_head) = np.unravel_index(max_entropy_idx, entropy.shape)
        print(f'Most distributed: Layer {ent_layer + 1}, Head {ent_head + 1} (entropy={entropy[ent_layer, ent_head]:.3f}')

    @staticmethod
    def print_head_comparison(data, layer, heads=None):
        if heads is None:
            heads = list(range(min(3, data.num_heads)))
        else:
            heads = [h for h in heads if 0 <= h < data.num_heads]
            if not heads:
                print('No valid head indices provided for comparison.')
                return
        print(f"\n{'=' * 60}")
        print(f'HEAD COMPARISON: Layer {layer + 1}, Heads {heads}')
        print(f"{'=' * 60}")
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
                    for (h, score) in scores[:2]:
                        print(f'H{h + 1}={score:.2f} ', end='')
                    print()

class HeadAnalyzer:

    def __init__(self, model):
        self.model = model
        self.hidden_dim = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads

    def get_qkv_matrices(self, data, layer, head):
        if not 0 <= layer < data.num_layers:
            raise IndexError(f'Layer index {layer} out of range (0-{data.num_layers - 1})')
        if not 0 <= head < data.num_heads:
            raise IndexError(f'Head index {head} out of range (0-{data.num_heads - 1})')
        hidden_state = data.hidden_states[layer]
        attn_layer = self.model.encoder.layer[layer].attention.self
        Q = attn_layer.query(hidden_state)
        K = attn_layer.key(hidden_state)
        V = attn_layer.value(hidden_state)
        (batch_size, seq_len) = hidden_state.shape[:2]
        for (matrix, name) in [(Q, 'Q'), (K, 'K'), (V, 'V')]:
            matrix = matrix.view(batch_size, seq_len, self.num_heads, self.head_dim)
            matrix = matrix.transpose(1, 2)
            if name == 'Q':
                Q = matrix
            elif name == 'K':
                K = matrix
            else:
                V = matrix
        return {'Q': Q[0, head].detach().cpu().numpy(), 'K': K[0, head].detach().cpu().numpy(), 'V': V[0, head].detach().cpu().numpy()}

    def print_detailed_head_analysis(self, data, layer, head):
        qkv = self.get_qkv_matrices(data, layer, head)
        print(f"\n{'=' * 80}")
        print(f'DETAILED HEAD ANALYSIS: Layer {layer + 1}, Head {head + 1}')
        print(f"{'=' * 80}")
        print(f'Dimension: {self.head_dim} per head')
        total_tokens = len(data.tokens)
        display_limit = min(total_tokens, AttentionFormatter.MAX_DISPLAY_TOKENS)
        display_indices = list(range(display_limit))
        if total_tokens > display_limit:
            print(f'(displaying first {display_limit} tokens of {total_tokens})')
        for (matrix_name, matrix) in qkv.items():
            print(f'\n{matrix_name} Matrix (all dimensions):')
            print('-' * 60)
            print(f"{'Token':<15} {'Vector Values (first 10)':<50} ||{matrix_name}||")
            for token_idx in display_indices:
                token = data.tokens[token_idx]
                values = matrix[token_idx]
                norm = np.linalg.norm(values)
                values_str = ' '.join([f'{v:6.3f}' for v in values[:10]])
                print(f'{token:<15} {values_str:<50} {norm:.3f}')
        scores = np.matmul(qkv['Q'], qkv['K'].T) / np.sqrt(self.head_dim)
        attention_weights = data.attention_weights[layer][0, head].cpu().numpy()
        print(f'\nPre-softmax scores (Q·K^T/√d):')
        print('-' * 60)
        header_label = 'From\\To'
        print(f'{header_label:<15}', end='')
        for j in display_indices:
            print(f'{j:8d}', end='')
        print()
        for row_idx in display_indices:
            from_token = data.tokens[row_idx]
            print(f'{row_idx:3d} {from_token:<11}', end='')
            for col_idx in display_indices:
                print(f'{scores[row_idx, col_idx]:8.3f}', end='')
            print()
        print(f'\nPost-softmax attention (from model):')
        print('-' * 60)
        print(f'{header_label:<15}', end='')
        for j in display_indices:
            print(f'{j:8d}', end='')
        print()
        for row_idx in display_indices:
            from_token = data.tokens[row_idx]
            print(f'{row_idx:3d} {from_token:<11}', end='')
            for col_idx in display_indices:
                val = attention_weights[row_idx, col_idx]
                if val > 0.1:
                    print(f'\x1b[1m{val:8.3f}\x1b[0m', end='')
                else:
                    print(f'{val:8.3f}', end='')
            print()
        print(f'\nVector Similarities:')
        print('-' * 60)
        for (name, matrix) in qkv.items():
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            normalized = matrix / (norms + 1e-08)
            sim_matrix = np.matmul(normalized, normalized.T)
            np.fill_diagonal(sim_matrix, -1)
            print(f'\n{name} Vector Cosine Similarities:')
            print(f"{'Token1':<15} {'Token2':<15} {'Cosine Similarity'}")
            flat_indices = np.argsort(sim_matrix.flatten())[-11:][::-1]
            similar_pairs = [(divmod(idx, sim_matrix.shape[1]), sim_matrix.flatten()[idx]) for idx in flat_indices]
            for ((i, j), similarity) in similar_pairs:
                if i != j:
                    print(f'{data.tokens[i]:<15} {data.tokens[j]:<15} {similarity:.6f}')
        print(f'\nDimension Statistics:')
        print('-' * 60)
        for (name, matrix) in qkv.items():
            mean_per_dim = np.mean(matrix, axis=0)
            var_per_dim = np.var(matrix, axis=0)
            std_per_dim = np.std(matrix, axis=0)
            top_var_dims = np.argsort(var_per_dim)[-5:][::-1]
            print(f'\n{name} Matrix Dimension Analysis:')
            print(f"{'Dim':<5} {'Mean':<10} {'Variance':<10} {'Std Dev':<10}")
            for dim in top_var_dims:
                print(f'{dim:<5} {mean_per_dim[dim]:<10.6f} {var_per_dim[dim]:<10.6f} {std_per_dim[dim]:<10.6f}')
        flat_attention = attention_weights.flatten()
        zero_weights = np.sum(flat_attention < 1e-06)
        total_weights = flat_attention.size
        sparsity = zero_weights / total_weights
        print(f'\nAttention Sparsity Analysis:')
        print('-' * 60)
        print(f'Zero weights: {zero_weights}/{total_weights} ({sparsity * 100:.2f}%)')
        print(f'Non-zero weights: {total_weights - zero_weights}')
        print(f'Min attention weight: {flat_attention.min():.6f}')
        print(f'Max attention weight: {flat_attention.max():.6f}')
        print(f'Mean attention weight: {flat_attention.mean():.6f}')
        print(f'Median attention weight: {np.median(flat_attention):.6f}')
        print(f'\nValue Enumeration (sorted by magnitude):')
        print('-' * 60)
        sorted_indices = np.argsort(flat_attention)[::-1]
        print(f"{'Rank':<6} {'From':<6} {'To':<6} {'Weight':<10} {'Tokens'}")
        for (i, idx) in enumerate(sorted_indices[:20]):
            (from_idx, to_idx) = divmod(idx, attention_weights.shape[1])
            weight = flat_attention[idx]
            from_token = data.tokens[from_idx]
            to_token = data.tokens[to_idx]
            print(f'{i + 1:<6} {from_idx:<6} {to_idx:<6} {weight:<10.6f} {from_token} → {to_token}')

    def print_head_analysis(self, data, layer, head):
        qkv = self.get_qkv_matrices(data, layer, head)
        print(f"\n{'=' * 60}")
        print(f'HEAD INTERNALS: Layer {layer + 1}, Head {head + 1}')
        print(f"{'=' * 60}")
        print(f'Dimension: {self.head_dim} per head')
        for (matrix_name, matrix) in qkv.items():
            print(f'\n{matrix_name} Matrix (first 5 dims):')
            print('-' * 40)
            for (i, token) in enumerate(data.tokens[:5]):
                values = matrix[i, :5]
                norm = np.linalg.norm(matrix[i])
                print(f'{token:12} [{values[0]:5.2f} {values[1]:5.2f} {values[2]:5.2f} {values[3]:5.2f} {values[4]:5.2f}...] ||{matrix_name}||={norm:.2f}')
        scores = np.matmul(qkv['Q'], qkv['K'].T) / np.sqrt(self.head_dim)
        attention_weights = data.attention_weights[layer][0, head].cpu().numpy()
        print(f'\nPre-softmax scores (Q·K^T/√d):')
        print('-' * 40)
        header_label = 'From\\To'
        print(f'{header_label:<12}', end='')
        for j in range(min(5, len(data.tokens))):
            print(f'{j:6d}', end='')
        print('...')
        print(f'\nPost-softmax attention (from model):')
        print('-' * 40)
        for i in range(min(5, len(data.tokens))):
            print(f'{data.tokens[i]:12}', end='')
            for j in range(min(5, len(data.tokens))):
                val = attention_weights[i, j]
                if val > 0.1:
                    print(f'\x1b[1m{val:6.3f}\x1b[0m', end='')
                else:
                    print(f'{val:6.3f}', end='')
            print('...')
        print(f'\nVector Similarities:')
        print('-' * 40)
        for (name, matrix) in qkv.items():
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            normalized = matrix / (norms + 1e-08)
            sim_matrix = np.matmul(normalized, normalized.T)
            np.fill_diagonal(sim_matrix, -1)
            max_sim_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            max_sim = sim_matrix[max_sim_idx]
            print(f"{name}: Most similar pair = '{data.tokens[max_sim_idx[0]]}' & '{data.tokens[max_sim_idx[1]]}' (cos={max_sim:.3f})")
        print(f'\nDimension Statistics:')
        print('-' * 40)
        for (name, matrix) in qkv.items():
            var_per_dim = np.var(matrix, axis=0)
            top_var_dims = np.argsort(var_per_dim)[-3:][::-1]
            print(f'{name}: Most variable dims = {top_var_dims[0]}, {top_var_dims[1]}, {top_var_dims[2]} (var={var_per_dim[top_var_dims[0]]:.3f}, {var_per_dim[top_var_dims[1]]:.3f}, {var_per_dim[top_var_dims[2]]:.3f})')

class DataExporter:

    @staticmethod
    def export_to_json(data, filepath):
        export_data = {'text': data.text, 'tokens': data.tokens, 'attention_weights': [], 'hidden_states': []}
        for layer_idx in range(data.num_layers):
            layer_weights = data.attention_weights[layer_idx]
            layer_data = {'layer': layer_idx, 'heads': []}
            for head_idx in range(data.num_heads):
                head_data = {'head': head_idx, 'attention_matrix': layer_weights[0, head_idx].cpu().numpy().tolist()}
                layer_data['heads'].append(head_data)
            export_data['attention_weights'].append(layer_data)
        for (layer_idx, hidden_state) in enumerate(data.hidden_states):
            export_data['hidden_states'].append({'layer': layer_idx, 'hidden_state': hidden_state[0].cpu().numpy().tolist()})
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f'Exported attention data to {filepath}')

    @staticmethod
    def export_attention_to_csv(data, filepath, layer=None, head=None):
        layers = [layer] if layer is not None else range(data.num_layers)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['layer', 'head', 'from_token_idx', 'to_token_idx', 'from_token', 'to_token', 'attention_weight']
            writer.writerow(header)
            for l in layers:
                heads = [head] if head is not None else range(data.num_heads)
                for h in heads:
                    attention = data.attention_weights[l][0, h].cpu().numpy()
                    for (i, from_token) in enumerate(data.tokens):
                        for (j, to_token) in enumerate(data.tokens):
                            writer.writerow([l, h, i, j, from_token, to_token, attention[i, j]])
        print(f'Exported attention weights to {filepath}')

    @staticmethod
    def export_sparsity_analysis(data, filepath):
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['layer', 'head', 'total_weights', 'zero_weights', 'sparsity_ratio'])
            for layer_idx in range(data.num_layers):
                layer_weights = data.attention_weights[layer_idx]
                for head_idx in range(data.num_heads):
                    attention = layer_weights[0, head_idx].cpu().numpy()
                    flat_attention = attention.flatten()
                    total_weights = flat_attention.size
                    zero_weights = np.sum(flat_attention < 1e-06)
                    sparsity = zero_weights / total_weights
                    writer.writerow([layer_idx, head_idx, total_weights, zero_weights, sparsity])
        print(f'Exported sparsity analysis to {filepath}')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('text', help='Text to analyze')
    parser.add_argument('-l', '--layer', type=int, default=0, help='Layer index (0-based)')
    parser.add_argument('-H', '--head', type=int, default=0, help='Head index (0-based)')
    parser.add_argument('--view', choices=['matrix', 'tokens', 'focus', 'detailed-weights', 'summary', 'compare-heads', 'internals', 'none'], default='matrix', help='Select visualization mode')
    parser.add_argument('--target-index', type=int, metavar='TOKEN_IDX', help='Token index used when view=focus')
    parser.add_argument('--internals-detail', choices=['basic', 'detailed'], default='basic', help='Detail level for view=internals')
    parser.add_argument('--compare-heads', nargs='*', type=int, metavar='HEAD', help='Compare specific heads when view=compare-heads')
    parser.add_argument('--include-special', action='store_true', help='Include special tokens ([CLS], [SEP])')
    parser.add_argument('--model', default='bert-base-uncased', help='Model name (default: bert-base-uncased)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Execution device')
    parser.add_argument('--export', metavar='KIND:PATH', help='Export data to file (kind: json, csv, sparsity)')
    args = parser.parse_args()
    try:
        visualizer = BERTAttentionVisualizer(args.model, args.device)
    except (RuntimeError, ValueError) as exc:
        print(f'Error: {exc}')
        return
    formatter = AttentionFormatter()
    data = visualizer.analyze_text(args.text)
    num_layers = data.num_layers
    num_heads = data.num_heads

    def validate_layer(layer_idx):
        if not 0 <= layer_idx < num_layers:
            print(f'Error: Layer index {layer_idx} out of range (0-{num_layers - 1})')
            return False
        return True

    def validate_head(head_idx):
        if not 0 <= head_idx < num_heads:
            print(f'Error: Head index {head_idx} out of range (0-{num_heads - 1})')
            return False
        return True
    view = args.view

    def handle_tokens():
        formatter.print_tokens(data.tokens)

    def handle_focus():
        index = args.target_index
        if index is None:
            print('Error: --target-index is required when view=focus')
            return
        if not 0 <= index < len(data.tokens):
            print(f'Error: Token index {index} out of range (0-{len(data.tokens) - 1})')
            return
        formatter.print_tokens(data.tokens, highlight_idx=index)
        formatter.print_attention_patterns(data, index)

    def handle_summary():
        formatter.print_attention_summary(data)

    def handle_compare_heads():
        if not validate_layer(args.layer):
            return
        heads = args.compare_heads if args.compare_heads else None
        formatter.print_head_comparison(data, args.layer, heads)

    def handle_internals():
        if not (validate_layer(args.layer) and validate_head(args.head)):
            return
        analyzer = HeadAnalyzer(visualizer.model)
        if args.internals_detail == 'detailed':
            analyzer.print_detailed_head_analysis(data, args.layer, args.head)
            return
        analyzer.print_head_analysis(data, args.layer, args.head)

    def handle_detailed_weights():
        if not (validate_layer(args.layer) and validate_head(args.head)):
            return
        formatter.print_detailed_attention_weights(data, args.layer, args.head)

    def handle_matrix():
        if not (validate_layer(args.layer) and validate_head(args.head)):
            return
        attention, tokens = visualizer.get_attention_matrix(data, args.layer, args.head, not args.include_special)
        formatter.print_attention_matrix(attention, tokens, args.layer, args.head, num_layers, num_heads)

    view_handlers = {
        'tokens': handle_tokens,
        'focus': handle_focus,
        'summary': handle_summary,
        'compare-heads': handle_compare_heads,
        'internals': handle_internals,
        'detailed-weights': handle_detailed_weights,
        'matrix': handle_matrix,
        'none': lambda: None,
    }

    view_handlers.get(view, handle_matrix)()

    def process_export(arg):
        if arg is None:
            return
        try:
            kind, path = arg.split(':', 1)
        except ValueError:
            print('Error')
            return
        actions = {
            'json': lambda: DataExporter().export_to_json(data, path),
            'csv': lambda: DataExporter().export_attention_to_csv(data, path, args.layer, args.head) if validate_layer(args.layer) and validate_head(args.head) else None,
            'sparsity': lambda: DataExporter().export_sparsity_analysis(data, path),
        }
        action = actions.get(kind.lower())
        if action:
            action()
        else:
            print(f"Error")

    process_export(args.export)
if __name__ == '__main__':
    main()
