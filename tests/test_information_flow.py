
import numpy as np
import pytest
from scipy.linalg import eig
from bert_attention.core import information_flow

class TestComputeAttentionRollout:

    def test_rollout_manual_calculation_first_layer(self, real_attention_simple):
        A0 = real_attention_simple['attention'][0][0].cpu().numpy().mean(0)
        seq_len = A0.shape[0]

        A0_with_residual = 0.5 * A0 + 0.5 * np.eye(seq_len)

        row_sums = A0_with_residual.sum(axis=1, keepdims=True)
        A0_normalized = A0_with_residual / (row_sums + 1e-9)

        expected_rollout0 = A0_normalized

        computed = information_flow.compute_attention_rollout(real_attention_simple)

        assert np.allclose(expected_rollout0, computed['per_layer'][0], atol=1e-8), "Rollout layer 0 doesn't match manual calculation"

    def test_rollout_manual_calculation_second_layer(self, real_attention_simple):
        A0 = real_attention_simple['attention'][0][0].cpu().numpy().mean(0)
        A1 = real_attention_simple['attention'][1][0].cpu().numpy().mean(0)
        seq_len = A0.shape[0]

        A0 = 0.5 * A0 + 0.5 * np.eye(seq_len)
        A0 = A0 / (A0.sum(axis=1, keepdims=True) + 1e-9)
        rollout0 = A0

        A1 = 0.5 * A1 + 0.5 * np.eye(seq_len)
        A1 = A1 / (A1.sum(axis=1, keepdims=True) + 1e-9)
        rollout1 = A1 @ rollout0
        rollout1 = rollout1 / (rollout1.sum(axis=1, keepdims=True) + 1e-9)

        computed = information_flow.compute_attention_rollout(real_attention_simple)

        assert np.allclose(rollout1, computed['per_layer'][1], atol=1e-8), "Rollout layer 1 doesn't match manual calculation"

    def test_rollout_rows_sum_to_one(self, real_attention_simple):
        computed = information_flow.compute_attention_rollout(real_attention_simple)

        for layer_idx, rollout_matrix in enumerate(computed['per_layer']):
            row_sums = rollout_matrix.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-6), f"Layer {layer_idx} rollout rows don't sum to 1.0: {row_sums}"

    def test_rollout_shape_consistency(self, real_attention_simple):
        computed = information_flow.compute_attention_rollout(real_attention_simple)

        num_layers = real_attention_simple['num_layers']
        seq_len = len(real_attention_simple['tokens'])

        assert computed['rollout'].shape == (num_layers, seq_len, seq_len), f"Expected shape ({num_layers}, {seq_len}, {seq_len}), got {computed['rollout'].shape}"

        assert len(computed['per_layer']) == num_layers

        assert computed['final_rollout'].shape == (seq_len, seq_len)

    def test_rollout_with_without_residual(self, real_attention_simple):
        rollout_with = information_flow.compute_attention_rollout(
            real_attention_simple, add_residual=True
        )
        rollout_without = information_flow.compute_attention_rollout(
            real_attention_simple, add_residual=False
        )

        assert not np.allclose(
            rollout_with['final_rollout'],
            rollout_without['final_rollout'],
            atol=1e-6
        ), "Rollout with/without residual should produce different results"

    def test_rollout_token_consistency(self, real_attention_simple):
        computed = information_flow.compute_attention_rollout(real_attention_simple)

        expected_tokens = real_attention_simple['tokens']
        assert computed['tokens'] == expected_tokens

class TestComputeAttentionFlow:

    def test_flow_manual_calculation(self, real_attention_simple):
        num_layers = real_attention_simple['num_layers']
        attention_matrices = []

        for layer in range(num_layers):
            layer_attn = real_attention_simple['attention'][layer][0].cpu().numpy()
            avg_attn = layer_attn.mean(axis=0)
            avg_attn = avg_attn / (avg_attn.sum(axis=1, keepdims=True) + 1e-9)
            attention_matrices.append(avg_attn)

        source_idx, target_idx = 0, 2
        seq_len = attention_matrices[0].shape[0]

        distribution = np.zeros(seq_len)
        distribution[source_idx] = 1.0

        manual_flow_values = []
        for attn_matrix in attention_matrices:
            distribution = distribution @ attn_matrix
            total_flow = distribution.sum()
            if total_flow > 0:
                distribution = distribution / total_flow
            manual_flow_values.append(distribution[target_idx])

        computed = information_flow.compute_attention_flow(
            attention_matrices, source_idx=source_idx, target_idx=target_idx
        )

        assert np.allclose(manual_flow_values, computed['flow_per_layer'], atol=1e-8), f"Manual: {manual_flow_values}, Computed: {computed['flow_per_layer']}"

    def test_flow_bounds(self, real_attention_simple):
        num_layers = real_attention_simple['num_layers']
        attention_matrices = []

        for layer in range(num_layers):
            layer_attn = real_attention_simple['attention'][layer][0].cpu().numpy()
            avg_attn = layer_attn.mean(axis=0)
            avg_attn = avg_attn / (avg_attn.sum(axis=1, keepdims=True) + 1e-9)
            attention_matrices.append(avg_attn)

        result = information_flow.compute_attention_flow(
            attention_matrices, source_idx=0, target_idx=2
        )

        assert 0.0 <= result['max_flow'] <= 1.0, f"Max flow out of bounds: {result['max_flow']}"

        for flow in result['flow_per_layer']:
            assert 0.0 <= flow <= 1.0, f"Flow value out of bounds: {flow}"

    def test_flow_same_source_target(self, real_attention_simple):
        num_layers = real_attention_simple['num_layers']
        attention_matrices = []

        for layer in range(num_layers):
            layer_attn = real_attention_simple['attention'][layer][0].cpu().numpy()
            avg_attn = layer_attn.mean(axis=0)
            avg_attn = avg_attn / (avg_attn.sum(axis=1, keepdims=True) + 1e-9)
            attention_matrices.append(avg_attn)

        result = information_flow.compute_attention_flow(
            attention_matrices, source_idx=0, target_idx=0
        )

        assert result['max_flow'] > 0.0, f"Expected positive flow to self, got {result['max_flow']}"

        assert 0.0 <= result['max_flow'] <= 1.0

    def test_flow_bottleneck_detection(self, real_attention_simple):
        num_layers = real_attention_simple['num_layers']
        attention_matrices = []

        for layer in range(num_layers):
            layer_attn = real_attention_simple['attention'][layer][0].cpu().numpy()
            avg_attn = layer_attn.mean(axis=0)
            avg_attn = avg_attn / (avg_attn.sum(axis=1, keepdims=True) + 1e-9)
            attention_matrices.append(avg_attn)

        result = information_flow.compute_attention_flow(
            attention_matrices, source_idx=0, target_idx=3
        )

        assert isinstance(result['bottleneck_layers'], list)

        for bottleneck in result['bottleneck_layers']:
            assert 0 <= bottleneck < num_layers

    def test_flow_consistency(self, real_attention_simple):
        num_layers = real_attention_simple['num_layers']
        attention_matrices = []

        for layer in range(num_layers):
            layer_attn = real_attention_simple['attention'][layer][0].cpu().numpy()
            avg_attn = layer_attn.mean(axis=0)
            avg_attn = avg_attn / (avg_attn.sum(axis=1, keepdims=True) + 1e-9)
            attention_matrices.append(avg_attn)

        result1 = information_flow.compute_attention_flow(
            attention_matrices, source_idx=0, target_idx=2
        )
        result2 = information_flow.compute_attention_flow(
            attention_matrices, source_idx=0, target_idx=2
        )

        assert result1['max_flow'] == result2['max_flow']
        assert np.allclose(result1['flow_per_layer'], result2['flow_per_layer'])

class TestComputeMarkovSteadyState:

    def test_steady_state_eigenvalue_cross_validation(self, real_attention_simple):
        final_layer = real_attention_simple['num_layers'] - 1
        final_attn = real_attention_simple['attention'][final_layer][0].cpu().numpy()
        avg_attn = final_attn.mean(axis=0)

        attn_norm = avg_attn / (avg_attn.sum(axis=1, keepdims=True) + 1e-9)

        eigenvalues, eigenvectors = eig(attn_norm.T)

        idx = np.argmin(np.abs(eigenvalues - 1.0))
        manual_steady = np.real(eigenvectors[:, idx])
        manual_steady = np.abs(manual_steady) / np.abs(manual_steady).sum()

        computed = information_flow.compute_markov_steady_state(avg_attn)

        assert np.allclose(manual_steady, computed['steady_state'], atol=1e-6), f"Manual and computed steady states differ significantly"

    def test_steady_state_sums_to_one(self, real_attention_simple):
        final_layer = real_attention_simple['num_layers'] - 1
        final_attn = real_attention_simple['attention'][final_layer][0].cpu().numpy()
        avg_attn = final_attn.mean(axis=0)

        result = information_flow.compute_markov_steady_state(avg_attn)

        steady_state_sum = result['steady_state'].sum()
        assert abs(steady_state_sum - 1.0) < 1e-6, f"Steady state sum {steady_state_sum} != 1.0"

    def test_steady_state_non_negative(self, real_attention_simple):
        final_layer = real_attention_simple['num_layers'] - 1
        final_attn = real_attention_simple['attention'][final_layer][0].cpu().numpy()
        avg_attn = final_attn.mean(axis=0)

        result = information_flow.compute_markov_steady_state(avg_attn)

        assert np.all(result['steady_state'] >= 0), f"Negative values in steady state: {result['steady_state']}"

    def test_steady_state_convergence_flag(self, real_attention_simple):
        final_layer = real_attention_simple['num_layers'] - 1
        final_attn = real_attention_simple['attention'][final_layer][0].cpu().numpy()
        avg_attn = final_attn.mean(axis=0)

        result = information_flow.compute_markov_steady_state(avg_attn)

        assert 'convergence' in result
        assert isinstance(result['convergence'], (bool, np.bool_))

    def test_steady_state_mathematical_property(self, real_attention_simple):
        final_layer = real_attention_simple['num_layers'] - 1
        final_attn = real_attention_simple['attention'][final_layer][0].cpu().numpy()
        avg_attn = final_attn.mean(axis=0)

        attn_norm = avg_attn / (avg_attn.sum(axis=1, keepdims=True) + 1e-9)

        result = information_flow.compute_markov_steady_state(avg_attn)
        steady_state = result['steady_state']

        left_side = steady_state @ attn_norm
        right_side = steady_state

        assert np.allclose(left_side, right_side, atol=1e-5), f"Steady state doesn't satisfy π @ A = π"

    def test_steady_state_alias_token_importance(self, real_attention_simple):
        final_layer = real_attention_simple['num_layers'] - 1
        final_attn = real_attention_simple['attention'][final_layer][0].cpu().numpy()
        avg_attn = final_attn.mean(axis=0)

        result = information_flow.compute_markov_steady_state(avg_attn)

        assert 'steady_state' in result
        assert 'token_importance' in result
        assert np.array_equal(result['steady_state'], result['token_importance'])

class TestComputeEffectiveAttention:

    def test_effective_attention_rollout_method(self, real_attention_simple):
        result = information_flow.compute_effective_attention(
            real_attention_simple, method='rollout'
        )

        assert 'rollout' in result
        assert 'final_rollout' in result
        assert 'per_layer' in result

    def test_effective_attention_flow_method(self, real_attention_simple):
        result = information_flow.compute_effective_attention(
            real_attention_simple,
            method='flow',
            source_idx=0,
            target_idx=2
        )

        assert 'max_flow' in result
        assert 'flow_per_layer' in result
        assert 'bottleneck_layers' in result

    def test_effective_attention_markov_method(self, real_attention_simple):
        result = information_flow.compute_effective_attention(
            real_attention_simple,
            method='markov'
        )

        assert 'steady_state' in result
        assert 'token_importance' in result
        assert 'convergence' in result

    def test_effective_attention_invalid_method(self, real_attention_simple):
        with pytest.raises(ValueError, match="Unknown method"):
            information_flow.compute_effective_attention(
                real_attention_simple,
                method='invalid_method'
            )

    def test_effective_attention_consistency(self, real_attention_simple):
        direct_rollout = information_flow.compute_attention_rollout(real_attention_simple)
        wrapper_rollout = information_flow.compute_effective_attention(
            real_attention_simple, method='rollout'
        )

        assert np.allclose(
            direct_rollout['final_rollout'],
            wrapper_rollout['final_rollout'],
            atol=1e-9
        )

        final_layer = real_attention_simple['num_layers'] - 1
        final_attn = real_attention_simple['attention'][final_layer][0].cpu().numpy()
        avg_attn = final_attn.mean(axis=0)

        direct_markov = information_flow.compute_markov_steady_state(avg_attn)
        wrapper_markov = information_flow.compute_effective_attention(
            real_attention_simple, method='markov'
        )

        assert np.allclose(
            direct_markov['steady_state'],
            wrapper_markov['steady_state'],
            atol=1e-9
        )
