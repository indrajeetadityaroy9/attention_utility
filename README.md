# BERT Attention Visualizer
CLI tool for exploring tokenization, attention matrices, and head internals of BERT-family models.

- `--view {matrix,tokens,focus,detailed-weights,summary,compare-heads,internals,none}` – choose the presentation (default `matrix`).
- `--layer` / `--head` – zero-based selectors for the attention layer/head used in relevant views.
- `--include-special` – keep `[CLS]`, `[SEP]`, and padding tokens in attention outputs.
- `--target-index` – token index required when `--view focus`.
- `--internals-detail {basic,detailed}` – level of detail for `--view internals`.
- `--compare-heads` – list of head indices for `--view compare-heads`.
- `--device {auto,cpu,cuda}` – execution device (`auto` picks CUDA when available).
- `--export KIND:PATH` – write data (`json`, `csv`, or `sparsity`).

## Examples

```bash
# Attention matrix for layer 3, head 7 without special tokens
python bert_attention_visualizer.py "Transformers are powerful" --layer 2 --head 6 --view matrix

# Inspect how token 5 attends across layers
python bert_attention_visualizer.py "Natural language processing" --view focus --target-index 5

# Dump detailed Q/K/V internals to the terminal
python bert_attention_visualizer.py "Sequence analysis" --view internals --internals-detail detailed --layer 1 --head 0

# Export all attention weights from layer 0 to CSV
python bert_attention_visualizer.py "Attention is all you need" --export csv:layer0.csv --layer 0 --head 0
```
