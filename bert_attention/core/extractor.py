
import numpy as np
import pandas as pd
import json

def extract_attention_matrix(model_output, layer, head, exclude_special=False):
    num_layers = model_output['num_layers']
    num_heads = model_output['num_heads']

    if not 0 <= layer < num_layers:
        raise IndexError(f'Layer {layer} out of range (0-{num_layers-1})')
    if not 0 <= head < num_heads:
        raise IndexError(f'Head {head} out of range (0-{num_heads-1})')

    attention = model_output['attention'][layer][0, head].cpu().numpy()
    tokens = model_output['tokens']

    if exclude_special:
        special_token_ids = {101, 102, 0}
        input_ids = model_output['input_ids'][0].cpu().numpy()
        mask = ~np.isin(input_ids, list(special_token_ids))

        attention = attention[mask][:, mask]
        tokens = [t for t, m in zip(tokens, mask) if m]

    return {
        'attention': attention,
        'tokens': tokens,
        'layer': layer,
        'head': head
    }

def extract_all_attention(model_output, exclude_special=False):
    num_layers = model_output['num_layers']
    num_heads = model_output['num_heads']

    attention_data = {}
    for layer in range(num_layers):
        attention_data[layer] = {}
        for head in range(num_heads):
            result = extract_attention_matrix(model_output, layer, head, exclude_special)
            attention_data[layer][head] = result['attention']

    return {
        'attention': attention_data,
        'tokens': model_output['tokens'] if not exclude_special else result['tokens'],
        'num_layers': num_layers,
        'num_heads': num_heads
    }

def extract_qkv_matrices(model_output, bert_model, layer, head):
    num_layers = model_output['num_layers']
    num_heads = model_output['num_heads']

    if not 0 <= layer < num_layers:
        raise IndexError(f'Layer {layer} out of range (0-{num_layers-1})')
    if not 0 <= head < num_heads:
        raise IndexError(f'Head {head} out of range (0-{num_heads-1})')

    hidden_state = model_output['hidden_states'][layer]

    attn_layer = bert_model.model.encoder.layer[layer].attention.self

    Q = attn_layer.query(hidden_state)
    K = attn_layer.key(hidden_state)
    V = attn_layer.value(hidden_state)

    batch_size, seq_len, _ = hidden_state.shape
    head_dim = bert_model.model.config.hidden_size // num_heads

    def reshape_qkv(tensor):
        tensor = tensor.view(batch_size, seq_len, num_heads, head_dim)
        tensor = tensor.transpose(1, 2)
        return tensor[0, head].detach().cpu().numpy()

    Q_reshaped = reshape_qkv(Q)
    K_reshaped = reshape_qkv(K)
    V_reshaped = reshape_qkv(V)

    return {
        'Q': Q_reshaped,
        'K': K_reshaped,
        'V': V_reshaped,
        'tokens': model_output['tokens'],
        'layer': layer,
        'head': head
    }

def attention_to_dataframe(model_output, layer=None, head=None):
    layers = [layer] if layer is not None else range(model_output['num_layers'])
    heads = [head] if head is not None else range(model_output['num_heads'])
    tokens = model_output['tokens']

    records = []
    for l in layers:
        for h in heads:
            attention = model_output['attention'][l][0, h].cpu().numpy()
            for i, from_token in enumerate(tokens):
                for j, to_token in enumerate(tokens):
                    records.append({
                        'layer': l,
                        'head': h,
                        'from_token_idx': i,
                        'to_token_idx': j,
                        'from_token': from_token,
                        'to_token': to_token,
                        'attention_weight': float(attention[i, j])
                    })

    return pd.DataFrame(records)

def export_to_csv(data, filepath, layer=None, head=None):
    df = attention_to_dataframe(data, layer, head)
    df.to_csv(filepath, index=False)

def export_to_json(data, filepath):
    export_data = {
        'text': data['text'],
        'tokens': data['tokens'],
        'num_layers': data['num_layers'],
        'num_heads': data['num_heads'],
        'attention_weights': []
    }

    for layer in range(data['num_layers']):
        layer_data = {
            'layer': layer,
            'heads': []
        }
        for head in range(data['num_heads']):
            attention = data['attention'][layer][0, head].cpu().numpy()
            layer_data['heads'].append({
                'head': head,
                'attention_matrix': attention.tolist()
            })
        export_data['attention_weights'].append(layer_data)

    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)

def export_qkv_to_npz(qkv_data, filepath):
    np.savez(
        filepath,
        Q=qkv_data['Q'],
        K=qkv_data['K'],
        V=qkv_data['V'],
        tokens=np.array(qkv_data['tokens']),
        layer=qkv_data['layer'],
        head=qkv_data['head']
    )
