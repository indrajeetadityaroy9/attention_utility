import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import argparse

class BERTAttentionAnalyzer:
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        print(f"Loading BERT-base-uncased model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(
            self.model_name, 
            output_attentions=True,
            attn_implementation="eager"
        )
        self.model = self.model.to(device)
        self.model.eval()
        print(f"BERT-base-uncased loaded successfully on {device}")
        print("Model architecture: 12 layers, 12 attention heads, 768 hidden dimensions")
        print("\nNote: BERT adds [CLS] at start (index 0) and [SEP] at end of your text")
        
    def get_attention_weights(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        attention = outputs.attentions
        return {
            'tokens': tokens,
            'attention': attention,
            'input_ids': inputs['input_ids']
        }
        
    def print_tokens(self, tokens, highlight_index=None):
        print("\n" + "="*60)
        print("TOKENIZATION")
        print("="*60)
        print(f"{'Index':<10} {'Token':<20} {'Type':<15}")
        print("-" * 45)
        for i, token in enumerate(tokens):
            special = token in ['[CLS]', '[SEP]', '[PAD]']
            marker = "special" if special else "word"
            if highlight_index is not None and i == highlight_index:
                print(f"\033[1;32m{i:<10} {token:<20} {marker:<15} ← FOCUS TOKEN\033[0m")
            else:
                print(f"{i:<10} {token:<20} {marker:<15}")
        print("\nTip: Use --focus-token INDEX to analyze attention to any specific token")
        
    def print_attention_matrix(self, attention_data, layer=0, head=0, include_special=False):
        tokens = attention_data['tokens']
        attention_weights = attention_data['attention']
        attention = attention_weights[layer][0, head].cpu().numpy()
        if not include_special:
            input_ids = attention_data['input_ids'].cpu().numpy()[0]
            special_tokens = {101, 102, 0}
            mask = ~np.isin(input_ids, list(special_tokens))
            attention = attention[mask][:, mask]
            tokens = [t for t, m in zip(tokens, mask) if m]
        print(f"\n" + "="*60)
        print(f"ATTENTION MATRIX - Layer {layer+1}/12, Head {head+1}/12")
        print("="*60)
        print(f"{'From/To':<15}", end='')
        for token in tokens:
            print(f"{token[:8]:<10}", end='')
        print()
        print("-" * (15 + 10 * len(tokens)))
        for i, from_token in enumerate(tokens):
            print(f"{from_token[:12]:<15}", end='')
            for j, attention_score in enumerate(attention[i]):
                if attention_score > 0.1:
                    print(f"\033[1m{attention_score:.4f}\033[0m    ", end='')
                else:
                    print(f"{attention_score:.4f}    ", end='')
            print()
        print(f"\nStatistics for Layer {layer+1}, Head {head+1}:")
        print(f"  Max attention: {np.max(attention):.4f}")
        print(f"  Mean attention: {np.mean(attention):.4f}")
        print(f"  Min attention: {np.min(attention):.4f}")
        
    def print_attention_summary(self, attention_data, include_special=False):
        tokens = attention_data['tokens']
        attention_weights = attention_data['attention']
        attention_array = np.stack([layer.cpu().numpy() for layer in attention_weights])
        attention_array = attention_array.squeeze(1)
        if not include_special:
            input_ids = attention_data['input_ids'].cpu().numpy()[0]
            special_tokens = {101, 102, 0}
            mask = ~np.isin(input_ids, list(special_tokens))
            attention_array = attention_array[:, :, mask, :]
            attention_array = attention_array[:, :, :, mask]
        print("\n" + "="*60)
        print("ATTENTION SUMMARY STATISTICS")
        print("="*60)
        max_attentions = np.max(attention_array, axis=(-1, -2))
        avg_attentions = np.mean(attention_array, axis=(-1, -2))
        print("\nMax attention weight per layer/head:")
        print("-" * 40)
        print("      ", end='')
        for head in range(12):
            print(f"Head{head+1:<2} ", end='')
        print()
        for layer in range(12):
            print(f"L{layer+1:<2}: ", end='')
            for head in range(12):
                print(f"{max_attentions[layer, head]:.3f}  ", end='')
            print()
        print("\nAverage attention weight per layer/head:")
        print("-" * 40)
        print("      ", end='')
        for head in range(12):
            print(f"Head{head+1:<2} ", end='')
        print()
        for layer in range(12):
            print(f"L{layer+1:<2}: ", end='')
            for head in range(12):
                print(f"{avg_attentions[layer, head]:.3f}  ", end='')
            print()
        print("\nOverall Statistics:")
        print("-" * 40)
        print(f"Global max attention: {np.max(attention_array):.4f}")
        print(f"Global mean attention: {np.mean(attention_array):.4f}")
        print(f"Global min attention: {np.min(attention_array):.4f}")
        
    def analyze(self, text, layer=0, head=0, include_special=False):
        print(f"\nAnalyzing: '{text}'")
        print("="*60)
        attention_data = self.get_attention_weights(text)
        self.print_tokens(attention_data['tokens'])
        self.print_attention_matrix(attention_data, layer, head, include_special)
        self.print_attention_summary(attention_data, include_special)
        return attention_data
        
    def analyze_multiple_layers(self, text, include_special=False):
        print(f"\nAnalyzing: '{text}'")
        print("="*60)
        attention_data = self.get_attention_weights(text)
        self.print_tokens(attention_data['tokens'])
        print(f"\nShowing attention matrices for selected layers/heads (BERT-base: 12 layers, 12 heads)")
        layer_head_pairs = [
            (0, 0),
            (5, 5),
            (11, 11)
        ]
        for layer, head in layer_head_pairs:
            self.print_attention_matrix(attention_data, layer, head, include_special)
        self.print_attention_summary(attention_data, include_special)
        return attention_data
        
    def visualize_specific_token_attention(self, text, token_index, layer=None, include_special=False):
        attention_data = self.get_attention_weights(text)
        tokens = attention_data['tokens']
        self.print_tokens(tokens, highlight_index=token_index)
        if token_index >= len(tokens):
            print(f"\nError: Token index {token_index} out of range. Text has {len(tokens)} tokens.")
            print("Remember: Index 0 is [CLS], your first word starts at index 1")
            return
        print(f"\n" + "="*60)
        print(f"ATTENTION TO TOKEN: '{tokens[token_index]}' (index {token_index})")
        print("="*60)
        if layer is not None:
            attention_layer = attention_data['attention'][layer][0].cpu().numpy()
            avg_attention_to_token = np.mean(attention_layer[:, :, token_index], axis=0)
            print(f"\nAverage attention from all tokens to '{tokens[token_index]}' (Layer {layer+1}):")
            print("-" * 40)
            for i, (token, score) in enumerate(zip(tokens, avg_attention_to_token)):
                if score > 0.1:
                    print(f"{i:<3} {token:<15} \033[1m{score:.4f}\033[0m {'*' * int(score * 50)}")
                else:
                    print(f"{i:<3} {token:<15} {score:.4f} {'*' * int(score * 50)}")
        else:
            print(f"\nAverage attention to '{tokens[token_index]}' across all 12 heads per layer:")
            print("="*70)
            print("Color coding: \033[1;32mGreen\033[0m = high (>0.15), \033[1mBold\033[0m = medium (>0.10), Normal = low")
            print("="*70)
            for layer_idx in range(12):
                attention_layer = attention_data['attention'][layer_idx][0].cpu().numpy()
                avg_attention_to_token = np.mean(attention_layer[:, :, token_index], axis=0)
                print(f"\nLayer {layer_idx+1}:")
                print("-" * 40)
                for i, (token, score) in enumerate(zip(tokens, avg_attention_to_token)):
                    if not include_special and token in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                    bar_length = int(score * 30)
                    bar = '█' * bar_length
                    if score > 0.15:
                        print(f"{i:<3} {token:<15} \033[1;32m{score:.4f}\033[0m {bar}")
                    elif score > 0.10:
                        print(f"{i:<3} {token:<15} \033[1m{score:.4f}\033[0m {bar}")
                    else:
                        print(f"{i:<3} {token:<15} {score:.4f} {bar}")
                max_idx = np.argmax(avg_attention_to_token)
                print(f"    → Highest: {tokens[max_idx]} ({avg_attention_to_token[max_idx]:.4f})")
                print(f"    → Average: {np.mean(avg_attention_to_token):.4f}")
            print("\n" + "="*70)
            print("SUMMARY ACROSS ALL LAYERS:")
            print("="*70)
            all_layer_attention = []
            for layer_idx in range(12):
                attention_layer = attention_data['attention'][layer_idx][0].cpu().numpy()
                avg_attention = np.mean(attention_layer[:, :, token_index], axis=0)
                all_layer_attention.append(avg_attention)
            all_layer_attention = np.array(all_layer_attention)
            mean_attention_per_token = np.mean(all_layer_attention, axis=0)
            sorted_indices = np.argsort(mean_attention_per_token)[::-1]
            print(f"\nTokens with highest average attention to '{tokens[token_index]}' across all layers:")
            print("-" * 50)
            for idx in sorted_indices[:5]:
                score = mean_attention_per_token[idx]
                print(f"{tokens[idx]:<15} {score:.4f} {'█' * int(score * 30)}")
            print(f"\n\nLayer-by-layer attention progression for top token '{tokens[sorted_indices[0]]}':")
            print("-" * 50)
            print("Layer:  ", end='')
            for i in range(12):
                print(f"{i+1:>5}", end='')
            print("\nScore:  ", end='')
            for i in range(12):
                score = all_layer_attention[i][sorted_indices[0]]
                print(f"{score:>5.3f}", end='')
            print()

def main():
    parser = argparse.ArgumentParser(
        description='Analyze attention patterns in BERT-base-uncased model',
        epilog='Example: python bert_attention_visualizer.py --text "Paris is the capital" --focus-token 1 (shows all 12 layers)'
    )
    parser.add_argument('--text', type=str, required=True,
                        help='Input text to analyze')
    parser.add_argument('--layer', type=int, default=None,
                        help='Layer index to visualize (0-11 for BERT-base). For --focus-token, omit to see all layers')
    parser.add_argument('--head', type=int, default=0,
                        help='Attention head index to visualize (0-11 for BERT-base)')
    parser.add_argument('--include-special', action='store_true',
                        help='Include [CLS], [SEP], and [PAD] tokens in visualization')
    parser.add_argument('--all-layers', action='store_true',
                        help='Show attention for multiple layers')
    parser.add_argument('--focus-token', type=int, default=None,
                        help='Show attention to a specific token index. Shows all 12 layers by default, or specific layer if --layer is used')
    parser.add_argument('--show-tokens-only', action='store_true',
                        help='Only show tokenization with indices (useful for finding token positions)')
    args = parser.parse_args()
    if args.layer is not None and (args.layer < 0 or args.layer > 11):
        print("Error: Layer must be between 0 and 11 for BERT-base")
        return
    if args.head < 0 or args.head > 11:
        print("Error: Head must be between 0 and 11 for BERT-base")
        return
    analyzer = BERTAttentionAnalyzer()
    if args.show_tokens_only:
        attention_data = analyzer.get_attention_weights(args.text)
        analyzer.print_tokens(attention_data['tokens'])
        return
    if args.focus_token is not None:
        analyzer.visualize_specific_token_attention(args.text, args.focus_token, args.layer, args.include_special)
    elif args.all_layers:
        analyzer.analyze_multiple_layers(args.text, args.include_special)
    else:
        layer = args.layer if args.layer is not None else 0
        analyzer.analyze(args.text, layer, args.head, args.include_special)

if __name__ == "__main__":
    main()