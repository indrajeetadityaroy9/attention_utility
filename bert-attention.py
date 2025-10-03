import argparse
import json
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from bert_attention.core.model import BERTModel
from bert_attention.core import extractor, metrics, information_flow


def compute_metrics_from_df(df, args):
    metric_list = args.with_metrics if hasattr(args, 'with_metrics') else args.metrics

    if 'all' in metric_list:
        metric_list = ['entropy', 'distance', 'sparsity', 'special-ratio']

    results = {}

    if args.per_head:
        per_head_results = []
        for (layer, head), group in df.groupby(['layer', 'head']):
            head_metrics = {}
            head_metrics['layer'] = int(layer)
            head_metrics['head'] = int(head)

            for metric_name in metric_list:
                head_metrics.update(compute_single_metric(group, metric_name, args))

            per_head_results.append(head_metrics)

        results = {
            'metadata': {
                'num_layers': int(df['layer'].max() + 1),
                'num_heads': int(df['head'].max() + 1),
                'total_records': len(df)
            },
            'metrics_per_head': per_head_results
        }
    else:
        for metric_name in metric_list:
            results.update(compute_single_metric(df, metric_name, args))

    return results


def compute_single_metric(df, metric_name, args):
    result = {}

    if metric_name == 'entropy':
        entropy_dict = metrics.compute_entropy(df)
        if isinstance(entropy_dict, dict):
            avg_entropy = sum(entropy_dict.values()) / len(entropy_dict)
            result['entropy'] = float(avg_entropy)
        else:
            result['entropy'] = float(entropy_dict)

    elif metric_name == 'special-ratio':
        special_tokens = args.special_tokens.split(',') if hasattr(args, 'special_tokens') and args.special_tokens else None
        ratio = metrics.compute_special_token_ratio(df, special_tokens)
        result['special_ratio'] = float(ratio)

    elif metric_name == 'distance':
        dist = metrics.compute_attention_distance(df)
        result['distance'] = float(dist)

    elif metric_name == 'sparsity':
        threshold = args.sparsity_threshold if hasattr(args, 'sparsity_threshold') else 1e-6
        sp = metrics.compute_sparsity(df, threshold=threshold)
        result['sparsity'] = float(sp)

    elif metric_name == 'flow':
        if not (hasattr(args, 'flow_source') and hasattr(args, 'flow_target')):
            raise ValueError("Flow metric requires --flow-source and --flow-target")

        attention_matrices = []
        for layer in sorted(df['layer'].unique()):
            layer_df = df[df['layer'] == layer]

            if 'head' in layer_df.columns:
                pivot = layer_df.groupby(['from_token_idx', 'to_token_idx'])['attention_weight'].mean().reset_index()
            else:
                pivot = layer_df

            seq_len = max(pivot['from_token_idx'].max(), pivot['to_token_idx'].max()) + 1
            matrix = np.zeros((seq_len, seq_len))

            for _, row in pivot.iterrows():
                i, j = int(row['from_token_idx']), int(row['to_token_idx'])
                matrix[i, j] = row['attention_weight']

            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = matrix / (row_sums + 1e-9)
            attention_matrices.append(matrix)

        flow_result = information_flow.compute_attention_flow(
            attention_matrices,
            source_idx=args.flow_source,
            target_idx=args.flow_target
        )
        result['flow'] = flow_result

    elif metric_name == 'markov':
        layer = args.markov_layer if hasattr(args, 'markov_layer') and args.markov_layer is not None else None

        if layer is not None:
            layer_df = df[df['layer'] == layer]
        else:
            layer_df = df

        if 'head' in layer_df.columns:
            pivot = layer_df.groupby(['from_token_idx', 'to_token_idx'])['attention_weight'].mean().reset_index()
        else:
            pivot = layer_df

        seq_len = max(pivot['from_token_idx'].max(), pivot['to_token_idx'].max()) + 1
        attention_matrix = np.zeros((seq_len, seq_len))

        for _, row in pivot.iterrows():
            i, j = int(row['from_token_idx']), int(row['to_token_idx'])
            attention_matrix[i, j] = row['attention_weight']

        markov_result = information_flow.compute_markov_steady_state(attention_matrix)

        tokens = []
        if 'from_token' in df.columns:
            token_df = df[['from_token_idx', 'from_token']].drop_duplicates().sort_values('from_token_idx')
            tokens = token_df['from_token'].tolist()

        result['markov'] = {
            'steady_state': markov_result['steady_state'].tolist(),
            'convergence': bool(markov_result['convergence']),
            'layer': layer if layer is not None else 'all'
        }

        if tokens:
            result['markov']['token_importance'] = [
                {'token': token, 'importance': float(importance)}
                for token, importance in zip(tokens, markov_result['steady_state'])
            ]

    return result


def save_output(data, filepath, format='json'):
    filepath = Path(filepath)

    if isinstance(data, pd.DataFrame):
        if format == 'csv':
            data.to_csv(filepath, index=False)
        elif format == 'json':
            data.to_json(filepath, orient='records', indent=2)
        elif format == 'parquet':
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format for DataFrame: {format}")
    else:
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'csv':
            df = pd.DataFrame([data]) if not isinstance(data, list) else pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


def handle_output_with_metrics(df, args, extraction_basename='data'):
    output_path = Path(args.output)

    metrics_result = compute_metrics_from_df(df, args)

    if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
        output_path.mkdir(parents=True, exist_ok=True)

        extraction_file = output_path / f"{extraction_basename}.csv"
        df.to_csv(extraction_file, index=False)
        print(f"Saved extraction data to {extraction_file}")

        metrics_format = args.metrics_format if hasattr(args, 'metrics_format') else 'json'
        metrics_ext = '.json' if metrics_format == 'json' else '.csv'
        metrics_file = output_path / f"metrics{metrics_ext}"
        save_output(metrics_result, metrics_file, metrics_format)
        print(f"Saved metrics to {metrics_file}")

    else:
        if hasattr(args, 'save_extraction') and args.save_extraction:
            df.to_csv(args.save_extraction, index=False)
            print(f"Saved extraction data to {args.save_extraction}")

        metrics_format = args.metrics_format if hasattr(args, 'metrics_format') else 'json'
        save_output(metrics_result, output_path, metrics_format)
        print(f"Saved metrics to {output_path}")


def cmd_attention(args):
    model = BERTModel(args.model, args.device)
    output = model.forward(args.text)

    if args.all_layers and args.all_heads:
        df = extractor.attention_to_dataframe(output)
    elif args.all_layers:
        df = extractor.attention_to_dataframe(output, head=args.head)
    elif args.all_heads:
        df = extractor.attention_to_dataframe(output, layer=args.layer)
    else:
        df = extractor.attention_to_dataframe(output, layer=args.layer, head=args.head)

    if hasattr(args, 'with_metrics') and args.with_metrics:
        handle_output_with_metrics(df, args, extraction_basename='attention')
    else:
        if args.format == 'csv':
            df.to_csv(args.output, index=False)
            print(f"Exported {len(df)} attention weights to {args.output}")
        elif args.format == 'json':
            df.to_json(args.output, orient='records', indent=2)
            print(f"Exported {len(df)} attention weights to {args.output}")
        elif args.format == 'parquet':
            df.to_parquet(args.output, index=False)
            print(f"Exported {len(df)} attention weights to {args.output}")


def cmd_tokens(args):
    model = BERTModel(args.model, args.device)
    output = model.forward(args.text)

    tokens_data = {
        'text': args.text,
        'tokens': output['tokens'],
        'num_tokens': len(output['tokens'])
    }

    if args.format == 'json':
        with open(args.output, 'w') as f:
            json.dump(tokens_data, f, indent=2)
        print(f"Exported {len(output['tokens'])} tokens to {args.output}")
    elif args.format == 'txt':
        with open(args.output, 'w') as f:
            for i, token in enumerate(output['tokens']):
                f.write(f"{i}\t{token}\n")
        print(f"Exported {len(output['tokens'])} tokens to {args.output}")


def cmd_qkv(args):
    model = BERTModel(args.model, args.device)
    output = model.forward(args.text)

    qkv_data = extractor.extract_qkv_matrices(output, model, args.layer, args.head)

    if args.format == 'npz':
        extractor.export_qkv_to_npz(qkv_data, args.output)
        print(f"Exported Q/K/V matrices for layer {args.layer}, head {args.head} to {args.output}")


def cmd_rollout(args):
    model = BERTModel(args.model, args.device)
    output = model.forward(args.text)

    rollout_data = information_flow.compute_attention_rollout(
        output,
        exclude_special=not args.include_special,
        add_residual=args.add_residual
    )

    tokens = rollout_data['tokens']

    if hasattr(args, 'with_metrics') and args.with_metrics:
        records = []
        for layer_idx, rollout_matrix in enumerate(rollout_data['per_layer']):
            for i, from_token in enumerate(tokens):
                for j, to_token in enumerate(tokens):
                    records.append({
                        'layer': layer_idx,
                        'from_token_idx': i,
                        'to_token_idx': j,
                        'from_token': from_token,
                        'to_token': to_token,
                        'attention_weight': float(rollout_matrix[i, j])
                    })

        df = pd.DataFrame(records)
        handle_output_with_metrics(df, args, extraction_basename='rollout')

    else:
        if args.format == 'csv':
            records = []
            for layer_idx, rollout_matrix in enumerate(rollout_data['per_layer']):
                for i, from_token in enumerate(tokens):
                    for j, to_token in enumerate(tokens):
                        records.append({
                            'layer': layer_idx,
                            'from_token_idx': i,
                            'to_token_idx': j,
                            'from_token': from_token,
                            'to_token': to_token,
                            'rollout_weight': float(rollout_matrix[i, j])
                        })

            df = pd.DataFrame(records)
            df.to_csv(args.output, index=False)
            print(f"Exported {len(records)} rollout weights to {args.output}")

        elif args.format == 'json':
            export_data = {
                'text': output['text'],
                'tokens': tokens,
                'num_layers': rollout_data['num_layers'],
                'rollout_per_layer': [r.tolist() for r in rollout_data['per_layer']],
                'final_rollout': rollout_data['final_rollout'].tolist()
            }
            with open(args.output, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Exported rollout data to {args.output}")

        elif args.format == 'npz':
            np.savez(
                args.output,
                rollout=rollout_data['rollout'],
                final_rollout=rollout_data['final_rollout'],
                tokens=np.array(tokens)
            )
            print(f"Exported rollout data to {args.output}")


def cmd_all(args):
    model = BERTModel(args.model, args.device)
    output = model.forward(args.text)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attention_file = output_dir / 'attention.csv'
    df = extractor.attention_to_dataframe(output)
    df.to_csv(attention_file, index=False)
    print(f"Exported attention weights to {attention_file}")

    tokens_file = output_dir / 'tokens.json'
    tokens_data = {
        'text': args.text,
        'tokens': output['tokens'],
        'num_tokens': len(output['tokens'])
    }
    with open(tokens_file, 'w') as f:
        json.dump(tokens_data, f, indent=2)
    print(f"Exported tokens to {tokens_file}")

    metadata_file = output_dir / 'metadata.json'
    config = model.get_config()
    config['text'] = args.text
    config['num_tokens'] = len(output['tokens'])
    with open(metadata_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Exported metadata to {metadata_file}")

    print(f"\nAll data exported to {output_dir}")


def cmd_analyze(args):
    input_files = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            input_files.extend(matches)
        else:
            input_files.append(pattern)

    if len(input_files) == 1:
        df = pd.read_csv(input_files[0])
    else:
        dfs = [pd.read_csv(f) for f in input_files]
        df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(df)} records from {len(input_files)} file(s)")

    metrics_result = compute_metrics_from_df(df, args)

    output_format = args.format if hasattr(args, 'format') else 'json'

    if output_format == 'json':
        with open(args.output, 'w') as f:
            json.dump(metrics_result, f, indent=2)
        print(f"Saved metrics to {args.output}")
    elif output_format == 'csv':
        if args.per_head and 'metrics_per_head' in metrics_result:
            df_out = pd.DataFrame(metrics_result['metrics_per_head'])
            df_out.to_csv(args.output, index=False)
            print(f"Saved metrics for {len(df_out)} heads to {args.output}")
        else:
            df_out = pd.DataFrame([metrics_result])
            df_out.to_csv(args.output, index=False)
            print(f"Saved metrics to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', default='bert-base-uncased',
                       help='HuggingFace model name (default: bert-base-uncased)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Compute device (default: auto)')

    subparsers = parser.add_subparsers(dest='command', required=True)

    attention_parser = subparsers.add_parser('attention',
        help='Extract attention matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    attention_parser.add_argument('text', help='Input text')
    attention_parser.add_argument('--layer', type=int, default=0, help='Layer index (0-based, default: 0)')
    attention_parser.add_argument('--head', type=int, default=0, help='Head index (0-based, default: 0)')
    attention_parser.add_argument('--all-layers', action='store_true', help='Extract all layers')
    attention_parser.add_argument('--all-heads', action='store_true', help='Extract all heads')
    attention_parser.add_argument('--format', default='csv', choices=['csv', 'json', 'parquet'],
                                  help='Output format (default: csv)')
    attention_parser.add_argument('--output', '-o', required=True,
                                  help='Output file or directory')

    metrics_group = attention_parser.add_argument_group('Metrics Pipeline')
    metrics_group.add_argument('--with-metrics', nargs='+',
                              choices=['entropy', 'distance', 'sparsity', 'special-ratio', 'flow', 'markov', 'all'],
                              help='Compute metrics: entropy, distance, sparsity, special-ratio, flow, markov, all')
    metrics_group.add_argument('--per-head', action='store_true',
                              help='Compute metrics per layer/head')
    metrics_group.add_argument('--save-extraction', metavar='PATH',
                              help='Save extraction data when using --with-metrics')
    metrics_group.add_argument('--metrics-format', choices=['json', 'csv'], default='json',
                              help='Format for metrics output (default: json)')

    metric_params = attention_parser.add_argument_group('Metric Parameters')
    metric_params.add_argument('--sparsity-threshold', type=float, default=1e-6,
                              help='Threshold for sparsity metric (default: 1e-6)')
    metric_params.add_argument('--special-tokens', default='[CLS],[SEP],[PAD]',
                              help='Comma-separated special tokens (default: [CLS],[SEP],[PAD])')
    metric_params.add_argument('--flow-source', type=int,
                              help='Source token index for flow metric')
    metric_params.add_argument('--flow-target', type=int,
                              help='Target token index for flow metric')
    metric_params.add_argument('--markov-layer', type=int,
                              help='Specific layer for markov metric (default: average all)')

    tokens_parser = subparsers.add_parser('tokens', help='Extract tokens')
    tokens_parser.add_argument('text', help='Input text')
    tokens_parser.add_argument('--format', default='json', choices=['json', 'txt'],
                              help='Output format (default: json)')
    tokens_parser.add_argument('--output', '-o', required=True, help='Output file path')

    qkv_parser = subparsers.add_parser('qkv', help='Extract Q/K/V matrices')
    qkv_parser.add_argument('text', help='Input text')
    qkv_parser.add_argument('--layer', type=int, required=True, help='Layer index (0-based)')
    qkv_parser.add_argument('--head', type=int, required=True, help='Head index (0-based)')
    qkv_parser.add_argument('--format', default='npz', choices=['npz'],
                           help='Output format (default: npz)')
    qkv_parser.add_argument('--output', '-o', required=True, help='Output file path')

    rollout_parser = subparsers.add_parser('rollout', help='Extract attention rollout')
    rollout_parser.add_argument('text', help='Input text')
    rollout_parser.add_argument('--include-special', action='store_true',
                               help='Include special tokens ([CLS], [SEP])')
    rollout_parser.add_argument('--add-residual', action='store_true', default=True,
                               help='Add residual connections (default: True)')
    rollout_parser.add_argument('--format', default='csv', choices=['csv', 'json', 'npz'],
                               help='Output format (default: csv)')
    rollout_parser.add_argument('--output', '-o', required=True, help='Output file or directory')

    rollout_metrics = rollout_parser.add_argument_group('Metrics Pipeline')
    rollout_metrics.add_argument('--with-metrics', nargs='+',
                                choices=['entropy', 'distance', 'sparsity', 'special-ratio', 'flow', 'markov', 'all'],
                                help='Compute metrics from rollout data')
    rollout_metrics.add_argument('--per-head', action='store_true',
                                help='Compute metrics per layer')
    rollout_metrics.add_argument('--save-extraction', metavar='PATH',
                                help='Save rollout data when using --with-metrics')
    rollout_metrics.add_argument('--metrics-format', choices=['json', 'csv'], default='json',
                                help='Format for metrics output (default: json)')

    rollout_metric_params = rollout_parser.add_argument_group('Metric Parameters')
    rollout_metric_params.add_argument('--sparsity-threshold', type=float, default=1e-6,
                                      help='Threshold for sparsity metric (default: 1e-6)')
    rollout_metric_params.add_argument('--special-tokens', default='[CLS],[SEP],[PAD]',
                                      help='Comma-separated special tokens')
    rollout_metric_params.add_argument('--flow-source', type=int,
                                      help='Source token index for flow metric')
    rollout_metric_params.add_argument('--flow-target', type=int,
                                      help='Target token index for flow metric')
    rollout_metric_params.add_argument('--markov-layer', type=int,
                                      help='Specific layer for markov metric')

    all_parser = subparsers.add_parser('all', help='Extract all data types')
    all_parser.add_argument('text', help='Input text')
    all_parser.add_argument('--output-dir', required=True, help='Output directory')

    analyze_parser = subparsers.add_parser('analyze',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    analyze_parser.add_argument('files', nargs='+', help='Input CSV file(s) or glob patterns')
    analyze_parser.add_argument('--metrics', nargs='+', required=True,
                               choices=['entropy', 'distance', 'sparsity', 'special-ratio', 'flow', 'markov', 'all'],
                               help='Metrics to compute')
    analyze_parser.add_argument('--output', '-o', required=True, help='Output file')
    analyze_parser.add_argument('--format', default='json', choices=['json', 'csv'],
                               help='Output format (default: json)')
    analyze_parser.add_argument('--per-head', action='store_true',
                               help='Compute metrics per layer/head')

    analyze_params = analyze_parser.add_argument_group('Metric Parameters')
    analyze_params.add_argument('--sparsity-threshold', type=float, default=1e-6,
                               help='Threshold for sparsity metric (default: 1e-6)')
    analyze_params.add_argument('--special-tokens', default='[CLS],[SEP],[PAD]',
                               help='Comma-separated special tokens')
    analyze_params.add_argument('--flow-source', type=int,
                               help='Source token index for flow metric')
    analyze_params.add_argument('--flow-target', type=int,
                               help='Target token index for flow metric')
    analyze_params.add_argument('--markov-layer', type=int,
                               help='Specific layer for markov metric')

    args = parser.parse_args()

    if hasattr(args, 'with_metrics') and args.with_metrics:
        if 'flow' in args.with_metrics:
            if not (hasattr(args, 'flow_source') and args.flow_source is not None and
                   hasattr(args, 'flow_target') and args.flow_target is not None):
                parser.error("Flow metric requires --flow-source and --flow-target")

    if hasattr(args, 'metrics') and args.metrics:
        if 'flow' in args.metrics:
            if not (hasattr(args, 'flow_source') and args.flow_source is not None and
                   hasattr(args, 'flow_target') and args.flow_target is not None):
                parser.error("Flow metric requires --flow-source and --flow-target")

    if args.command == 'attention':
        cmd_attention(args)
    elif args.command == 'tokens':
        cmd_tokens(args)
    elif args.command == 'qkv':
        cmd_qkv(args)
    elif args.command == 'rollout':
        cmd_rollout(args)
    elif args.command == 'all':
        cmd_all(args)
    elif args.command == 'analyze':
        cmd_analyze(args)


if __name__ == '__main__':
    main()
