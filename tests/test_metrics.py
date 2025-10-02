
import numpy as np
import pandas as pd
import pytest
from scipy.stats import entropy as scipy_entropy
from bert_attention.core import metrics, extractor

class TestComputeEntropy:

    def test_entropy_manual_calculation_numpy(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )
        attention_matrix = attention_data['attention']

        manual_entropies = []
        for i in range(attention_matrix.shape[0]):
            row = attention_matrix[i]
            row_with_epsilon = row + 1e-9
            row_normalized = row_with_epsilon / row_with_epsilon.sum()
            manual_entropy = -np.sum(row_normalized * np.log(row_normalized))
            manual_entropies.append(manual_entropy)

        manual_avg_entropy = np.mean(manual_entropies)

        computed_entropy = metrics.compute_entropy(attention_matrix)

        assert abs(manual_avg_entropy - computed_entropy) < 1e-9, f"Manual: {manual_avg_entropy}, Computed: {computed_entropy}"

    def test_entropy_scipy_cross_validation(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )
        attention_matrix = attention_data['attention']

        our_entropy = metrics.compute_entropy(attention_matrix)

        scipy_entropies = []
        for i in range(attention_matrix.shape[0]):
            row = attention_matrix[i] + 1e-9
            row = row / row.sum()
            scipy_entropies.append(scipy_entropy(row, base=np.e))

        scipy_avg = np.mean(scipy_entropies)

        assert abs(our_entropy - scipy_avg) < 1e-8, f"Our: {our_entropy}, Scipy: {scipy_avg}"

    def test_entropy_dataframe_input(self, real_attention_simple):
        df = extractor.attention_to_dataframe(
            real_attention_simple, layer=0, head=0
        )

        entropy_dict = metrics.compute_entropy(df)

        num_tokens = len(real_attention_simple['tokens'])
        assert len(entropy_dict) == num_tokens

        for idx, ent in entropy_dict.items():
            assert ent >= 0, f"Negative entropy at index {idx}: {ent}"

    def test_entropy_bounds(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )
        attention_matrix = attention_data['attention']
        seq_len = attention_matrix.shape[0]

        entropy = metrics.compute_entropy(attention_matrix)

        assert entropy >= 0, f"Negative entropy: {entropy}"

        max_entropy = np.log(seq_len)
        assert entropy <= max_entropy + 1e-6, f"Entropy {entropy} exceeds maximum {max_entropy}"

    def test_entropy_edge_case_single_token(self, real_model):
        output = real_model.forward("Hello")
        attention_data = extractor.extract_attention_matrix(output, layer=0, head=0)
        attention_matrix = attention_data['attention']

        entropy = metrics.compute_entropy(attention_matrix)

        assert entropy >= 0
        assert not np.isnan(entropy)

class TestComputeSpecialTokenRatio:

    def test_special_ratio_manual_calculation(self, real_attention_simple):
        df = extractor.attention_to_dataframe(
            real_attention_simple, layer=0, head=0
        )

        special_tokens = ['[CLS]', '[SEP]', '[PAD]']
        total_attention = df['attention_weight'].sum()
        special_attention = df[
            df['to_token'].isin(special_tokens)
        ]['attention_weight'].sum()
        manual_ratio = special_attention / total_attention

        computed_ratio = metrics.compute_special_token_ratio(df)

        assert abs(manual_ratio - computed_ratio) < 1e-12,f"Manual: {manual_ratio}, Computed: {computed_ratio}"

    def test_special_ratio_all_special_tokens(self, real_attention_special):
        df = extractor.attention_to_dataframe(
            real_attention_special, layer=0, head=0
        )

        ratio = metrics.compute_special_token_ratio(df)

        assert ratio > 0.9, f"Expected high ratio for all special tokens, got {ratio}"

    def test_special_ratio_bounds(self, real_attention_simple):
        df = extractor.attention_to_dataframe(
            real_attention_simple, layer=0, head=0
        )

        ratio = metrics.compute_special_token_ratio(df)

        assert 0.0 <= ratio <= 1.0, f"Ratio out of bounds: {ratio}"

    def test_special_ratio_custom_tokens(self, real_attention_simple):
        df = extractor.attention_to_dataframe(
            real_attention_simple, layer=0, head=0
        )

        custom_ratio = metrics.compute_special_token_ratio(df, special_tokens=['[CLS]'])

        total = df['attention_weight'].sum()
        cls_attention = df[df['to_token'] == '[CLS]']['attention_weight'].sum()
        expected = cls_attention / total

        assert abs(custom_ratio - expected) < 1e-12

    def test_special_ratio_dict_input(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )

        ratio_dict = metrics.compute_special_token_ratio(attention_data)
        assert 0.0 <= ratio_dict <= 1.0

class TestComputeAttentionDistance:

    def test_distance_manual_calculation_numpy(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )
        matrix = attention_data['attention']

        total_weighted_distance = 0.0
        total_weight = 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                distance = abs(i - j)
                weight = matrix[i, j]
                total_weighted_distance += distance * weight
                total_weight += weight

        manual_distance = total_weighted_distance / total_weight

        computed_distance = metrics.compute_attention_distance(matrix)

        assert abs(manual_distance - computed_distance) < 1e-9, f"Manual: {manual_distance}, Computed: {computed_distance}"

    def test_distance_manual_calculation_dataframe(self, real_attention_simple):
        df = extractor.attention_to_dataframe(
            real_attention_simple, layer=0, head=0
        )

        distances = np.abs(df['from_token_idx'] - df['to_token_idx'])
        weighted_distances = distances * df['attention_weight']
        manual_distance = weighted_distances.sum() / df['attention_weight'].sum()

        computed_distance = metrics.compute_attention_distance(df)

        assert abs(manual_distance - computed_distance) < 1e-9

    def test_distance_non_negative(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )

        distance = metrics.compute_attention_distance(attention_data['attention'])

        assert distance >= 0, f"Negative distance: {distance}"

    def test_distance_bounds(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )
        matrix = attention_data['attention']
        seq_len = matrix.shape[0]

        distance = metrics.compute_attention_distance(matrix)

        assert distance <= seq_len, f"Distance {distance} exceeds sequence length {seq_len}"

    def test_distance_numpy_vs_dataframe(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )
        df = extractor.attention_to_dataframe(
            real_attention_simple, layer=0, head=0
        )

        distance_numpy = metrics.compute_attention_distance(attention_data['attention'])
        distance_df = metrics.compute_attention_distance(df)

        assert abs(distance_numpy - distance_df) < 1e-9

class TestComputeSparsity:

    def test_sparsity_manual_calculation(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )
        matrix = attention_data['attention']

        threshold = 1e-6

        below_threshold = np.sum(matrix < threshold)
        total = matrix.size
        manual_sparsity = below_threshold / total

        computed_sparsity = metrics.compute_sparsity(matrix, threshold=threshold)

        assert abs(manual_sparsity - computed_sparsity) < 1e-12

    def test_sparsity_dataframe_input(self, real_attention_simple):
        df = extractor.attention_to_dataframe(
            real_attention_simple, layer=0, head=0
        )

        threshold = 1e-6
        weights = df['attention_weight'].values

        below_threshold = np.sum(weights < threshold)
        manual_sparsity = below_threshold / len(weights)

        computed_sparsity = metrics.compute_sparsity(df, threshold=threshold)

        assert abs(manual_sparsity - computed_sparsity) < 1e-12

    def test_sparsity_bounds(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )

        sparsity = metrics.compute_sparsity(attention_data['attention'])

        assert 0.0 <= sparsity <= 1.0, f"Sparsity out of bounds: {sparsity}"

    def test_sparsity_different_thresholds(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )
        matrix = attention_data['attention']

        sparsity_low = metrics.compute_sparsity(matrix, threshold=1e-9)
        sparsity_mid = metrics.compute_sparsity(matrix, threshold=1e-6)
        sparsity_high = metrics.compute_sparsity(matrix, threshold=1e-3)

        assert sparsity_low <= sparsity_mid <= sparsity_high, f"Sparsity not monotonic: {sparsity_low}, {sparsity_mid}, {sparsity_high}"

class TestComputeMaxAttention:

    def test_max_attention_manual_calculation(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )
        matrix = attention_data['attention']

        manual_max = np.max(matrix)

        computed_max = metrics.compute_max_attention(matrix)

        assert abs(manual_max - computed_max) < 1e-12

    def test_max_attention_dataframe_input(self, real_attention_simple):
        df = extractor.attention_to_dataframe(
            real_attention_simple, layer=0, head=0
        )

        manual_max = df['attention_weight'].max()
        computed_max = metrics.compute_max_attention(df)

        assert abs(manual_max - computed_max) < 1e-12

    def test_max_attention_bounds(self, real_attention_simple):
        attention_data = extractor.extract_attention_matrix(
            real_attention_simple, layer=0, head=0
        )

        max_attn = metrics.compute_max_attention(attention_data['attention'])

        assert 0.0 <= max_attn <= 1.0, f"Max attention out of bounds: {max_attn}"

class TestComputeAllMetrics:

    def test_all_metrics_completeness(self, real_attention_simple):
        df = extractor.attention_to_dataframe(
            real_attention_simple, layer=0, head=0
        )

        all_metrics = metrics.compute_all_metrics(df)

        expected_keys = ['entropy', 'special_ratio', 'distance', 'sparsity', 'max_attention']
        for key in expected_keys:
            assert key in all_metrics, f"Missing metric: {key}"
            assert all_metrics[key] is not None, f"Metric {key} is None"

    def test_all_metrics_individual_match(self, real_attention_simple):
        df = extractor.attention_to_dataframe(
            real_attention_simple, layer=0, head=0
        )

        all_metrics = metrics.compute_all_metrics(df)

        individual_entropy = metrics.compute_entropy(df)
        individual_special = metrics.compute_special_token_ratio(df)
        individual_distance = metrics.compute_attention_distance(df)
        individual_sparsity = metrics.compute_sparsity(df)
        individual_max = metrics.compute_max_attention(df)

        if isinstance(all_metrics['entropy'], dict):
            all_metrics['entropy'] = sum(all_metrics['entropy'].values()) / len(all_metrics['entropy'])
        if isinstance(individual_entropy, dict):
            individual_entropy = sum(individual_entropy.values()) / len(individual_entropy)

        assert abs(all_metrics['entropy'] - individual_entropy) < 1e-9
        assert abs(all_metrics['special_ratio'] - individual_special) < 1e-9
        assert abs(all_metrics['distance'] - individual_distance) < 1e-9
        assert abs(all_metrics['sparsity'] - individual_sparsity) < 1e-9
        assert abs(all_metrics['max_attention'] - individual_max) < 1e-9

    def test_layer_head_metrics(self, real_attention_simple):
        df = extractor.attention_to_dataframe(real_attention_simple)

        result_df = metrics.compute_layer_head_metrics(df)

        num_layers = real_attention_simple['num_layers']
        num_heads = real_attention_simple['num_heads']
        expected_rows = num_layers * num_heads

        assert len(result_df) == expected_rows

        required_cols = ['layer', 'head', 'entropy', 'special_ratio', 'distance', 'sparsity', 'max_attention']
        for col in required_cols:
            assert col in result_df.columns, f"Missing column: {col}"
