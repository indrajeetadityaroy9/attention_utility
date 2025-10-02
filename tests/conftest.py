
import pytest
import numpy as np
from bert_attention.core.model import BERTModel

@pytest.fixture(scope="session")
def real_model():
    return BERTModel('bert-base-uncased', device='cpu')

@pytest.fixture(scope="session")
def distilbert_model():
    return BERTModel('distilbert-base-uncased', device='cpu')

@pytest.fixture
def sample_texts():
    return {
        'simple_short': "The cat sat.",
        'medium': "The quick brown fox jumps.",
        'all_special': "[CLS] [SEP] [PAD]",
        'research_quote': "Attention is all you need.",
        'single_word': "Hello",
        'repeated_tokens': "the the the the",
        'long_sequence': "Natural language processing with transformers has revolutionized the field of artificial intelligence and machine learning."
    }

@pytest.fixture
def real_attention_simple(real_model):
    return real_model.forward("The cat sat.")

@pytest.fixture
def real_attention_medium(real_model):
    return real_model.forward("The quick brown fox jumps.")

@pytest.fixture
def real_attention_special(real_model):
    return real_model.forward("[CLS] [SEP] [PAD]")

@pytest.fixture
def softmax_func():
    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    return softmax

def assert_attention_properties(attention_matrix, tol=1e-6):
    assert np.all(attention_matrix >= 0), "Attention has negative values"
    assert np.all(attention_matrix <= 1), "Attention has values > 1"

    row_sums = attention_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=tol), f"Row sums not close to 1.0: {row_sums}"

def assert_probability_distribution(distribution, tol=1e-6):
    assert np.all(distribution >= 0), "Distribution has negative values"
    assert np.allclose(distribution.sum(), 1.0, atol=tol), f"Distribution does not sum to 1.0: {distribution.sum()}"
