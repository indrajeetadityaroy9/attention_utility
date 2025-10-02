
import numpy as np
from scipy.linalg import eig
import warnings

def compute_attention_rollout(model_output, exclude_special=False, add_residual=True):
    num_layers = model_output['num_layers']
    seq_len = len(model_output['tokens'])

    rollout_current = np.eye(seq_len)
    rollout_per_layer = []

    for layer in range(num_layers):
        layer_attention = model_output['attention'][layer][0].cpu().numpy()
        avg_attention = layer_attention.mean(axis=0)

        if add_residual:
            avg_attention = 0.5 * avg_attention + 0.5 * np.eye(seq_len)

        row_sums = avg_attention.sum(axis=1, keepdims=True)
        avg_attention = avg_attention / (row_sums + 1e-9)

        rollout_current = avg_attention @ rollout_current

        row_sums = rollout_current.sum(axis=1, keepdims=True)
        rollout_current = rollout_current / (row_sums + 1e-9)

        rollout_per_layer.append(rollout_current.copy())

    rollout_all = np.stack(rollout_per_layer, axis=0)

    tokens = model_output['tokens']
    if exclude_special:
        special_token_ids = {101, 102, 0}
        input_ids = model_output['input_ids'][0].cpu().numpy()
        mask = ~np.isin(input_ids, list(special_token_ids))

        rollout_all = rollout_all[:, mask][:, :, mask]
        tokens = [t for t, m in zip(tokens, mask) if m]

    return {
        'rollout': rollout_all,
        'per_layer': [rollout_all[i] for i in range(num_layers)],
        'final_rollout': rollout_all[-1],
        'tokens': tokens,
        'num_layers': num_layers
    }

def compute_attention_flow(attention_matrices, source_idx, target_idx):
    if not attention_matrices:
        raise ValueError("attention_matrices cannot be empty")

    num_layers = len(attention_matrices)
    seq_len = attention_matrices[0].shape[0]

    if not (0 <= source_idx < seq_len):
        raise IndexError(f"source_idx {source_idx} out of range [0, {seq_len-1}]")
    if not (0 <= target_idx < seq_len):
        raise IndexError(f"target_idx {target_idx} out of range [0, {seq_len-1}]")

    flow_per_layer = []
    current_distribution = np.zeros(seq_len)
    current_distribution[source_idx] = 1.0

    for layer_idx, attention in enumerate(attention_matrices):
        next_distribution = current_distribution @ attention

        total_flow = next_distribution.sum()
        if total_flow > 0:
            next_distribution = next_distribution / total_flow

        flow_to_target = next_distribution[target_idx]
        flow_per_layer.append(flow_to_target)

        current_distribution = next_distribution

    max_flow = flow_per_layer[-1] if flow_per_layer else 0.0

    bottleneck_layers = []
    for i in range(1, len(flow_per_layer)):
        if flow_per_layer[i] < 0.9 * flow_per_layer[i-1]:
            bottleneck_layers.append(i)

    return {
        'max_flow': float(max_flow),
        'flow_per_layer': [float(f) for f in flow_per_layer],
        'bottleneck_layers': bottleneck_layers,
        'source_idx': source_idx,
        'target_idx': target_idx
    }

def compute_markov_steady_state(attention_weights):
    if not isinstance(attention_weights, np.ndarray):
        attention_weights = np.array(attention_weights)

    seq_len = attention_weights.shape[0]

    row_sums = attention_weights.sum(axis=1, keepdims=True)
    attention_weights = attention_weights / (row_sums + 1e-9)

    try:
        eigenvalues, eigenvectors = eig(attention_weights.T)

        idx = np.argmin(np.abs(eigenvalues - 1.0))
        steady_state = np.real(eigenvectors[:, idx])

        steady_state = np.abs(steady_state)
        steady_state = steady_state / steady_state.sum()

        convergence = np.abs(eigenvalues[idx] - 1.0) < 1e-6

    except Exception as e:
        warnings.warn(f"Eigenvalue method failed: {e}. Using power iteration.")

        steady_state = np.ones(seq_len) / seq_len
        max_iter = 1000
        tolerance = 1e-6

        for iteration in range(max_iter):
            next_state = steady_state @ attention_weights

            if np.allclose(next_state, steady_state, atol=tolerance):
                convergence = True
                break

            steady_state = next_state
        else:
            convergence = False
            warnings.warn(f"Power iteration did not converge after {max_iter} iterations")

    return {
        'steady_state': steady_state,
        'token_importance': steady_state,
        'convergence': convergence
    }

def compute_effective_attention(model_output, method='rollout', **kwargs):
    if method == 'rollout':
        return compute_attention_rollout(model_output, **kwargs)

    elif method == 'flow':
        num_layers = model_output['num_layers']
        num_heads = model_output['num_heads']

        attention_matrices = []
        for layer in range(num_layers):
            layer_attn = model_output['attention'][layer][0].cpu().numpy()
            avg_attn = layer_attn.mean(axis=0)
            attention_matrices.append(avg_attn)

        return compute_attention_flow(attention_matrices, **kwargs)

    elif method == 'markov':
        final_layer = model_output['num_layers'] - 1
        final_attn = model_output['attention'][final_layer][0].cpu().numpy()
        avg_attn = final_attn.mean(axis=0)

        return compute_markov_steady_state(avg_attn)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: rollout, flow, markov")
