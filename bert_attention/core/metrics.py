
import numpy as np
import pandas as pd

def compute_entropy(attention_weights):
    if isinstance(attention_weights, pd.DataFrame):
        def _entropy(group):
            weights = group['attention_weight'].values
            weights = weights + 1e-9
            weights = weights / weights.sum()
            return -np.sum(weights * np.log(weights))

        return attention_weights.groupby('from_token_idx').apply(_entropy).to_dict()

    if len(attention_weights.shape) != 2:
        raise ValueError("attention_weights must be 2D array")

    entropies = []
    for i in range(attention_weights.shape[0]):
        weights = attention_weights[i] + 1e-9
        weights = weights / weights.sum()
        entropy = -np.sum(weights * np.log(weights))
        entropies.append(entropy)

    return np.mean(entropies)

def compute_special_token_ratio(attention_data, special_tokens=None):
    if special_tokens is None:
        special_tokens = ['[CLS]', '[SEP]', '[PAD]']

    if isinstance(attention_data, pd.DataFrame):
        total_attention = attention_data['attention_weight'].sum()
        special_attention = attention_data[
            attention_data['to_token'].isin(special_tokens)
        ]['attention_weight'].sum()
        return special_attention / total_attention if total_attention > 0 else 0.0

    if isinstance(attention_data, dict):
        attention = attention_data['attention']
        tokens = attention_data['tokens']

        total_attention = attention.sum()
        special_attention = 0.0

        for i in range(len(tokens)):
            if tokens[i] in special_tokens:
                special_attention += attention[:, i].sum()

        return special_attention / total_attention if total_attention > 0 else 0.0

    raise TypeError("attention_data must be DataFrame or dict")

def compute_attention_distance(attention_data):
    if isinstance(attention_data, pd.DataFrame):
        distances = np.abs(
            attention_data['from_token_idx'] - attention_data['to_token_idx']
        )
        weighted_distances = distances * attention_data['attention_weight']
        total_weight = attention_data['attention_weight'].sum()
        return weighted_distances.sum() / total_weight if total_weight > 0 else 0.0

    if isinstance(attention_data, np.ndarray):
        seq_len = attention_data.shape[0]
        total_weighted_distance = 0.0
        total_weight = 0.0

        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                weight = attention_data[i, j]
                total_weighted_distance += distance * weight
                total_weight += weight

        return total_weighted_distance / total_weight if total_weight > 0 else 0.0

    raise TypeError("attention_data must be DataFrame or numpy array")

def compute_sparsity(attention_weights, threshold=1e-6):
    if isinstance(attention_weights, pd.DataFrame):
        weights = attention_weights['attention_weight'].values
    elif isinstance(attention_weights, np.ndarray):
        weights = attention_weights.flatten()
    else:
        raise TypeError("attention_weights must be DataFrame or numpy array")

    zero_weights = np.sum(weights < threshold)
    total_weights = len(weights)

    return zero_weights / total_weights if total_weights > 0 else 0.0

def compute_max_attention(attention_weights):
    if isinstance(attention_weights, pd.DataFrame):
        return attention_weights['attention_weight'].max()
    elif isinstance(attention_weights, np.ndarray):
        return float(np.max(attention_weights))
    else:
        raise TypeError("attention_weights must be DataFrame or numpy array")

def compute_all_metrics(attention_data, include_entropy=True, include_special_ratio=True,
                       include_distance=True, include_sparsity=True, include_max=True):
    metrics = {}

    if include_entropy:
        try:
            metrics['entropy'] = compute_entropy(attention_data)
        except Exception as e:
            metrics['entropy'] = None

    if include_special_ratio:
        try:
            metrics['special_ratio'] = compute_special_token_ratio(attention_data)
        except Exception as e:
            metrics['special_ratio'] = None

    if include_distance:
        try:
            metrics['distance'] = compute_attention_distance(attention_data)
        except Exception as e:
            metrics['distance'] = None

    if include_sparsity:
        try:
            metrics['sparsity'] = compute_sparsity(attention_data)
        except Exception as e:
            metrics['sparsity'] = None

    if include_max:
        try:
            metrics['max_attention'] = compute_max_attention(attention_data)
        except Exception as e:
            metrics['max_attention'] = None

    return metrics

def compute_layer_head_metrics(dataframe):
    results = []

    for (layer, head), group in dataframe.groupby(['layer', 'head']):
        metrics = compute_all_metrics(group)
        metrics['layer'] = layer
        metrics['head'] = head
        results.append(metrics)

    return pd.DataFrame(results)
