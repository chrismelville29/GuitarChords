from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxnn.nn.layers import Conv2D, BatchNorm
from jaxnn.nn import losses, activations


def test_conv2d_init_shapes():
    layer = Conv2D(in_channels=3, out_channels=5, kernel_size=(3, 3))
    params = layer.init(jax.random.PRNGKey(0))
    assert params["w"].shape == (3, 3, 3, 5)
    assert params["b"].shape == (5,)


def test_conv2d_forward_valid_matches_manual_sum():
    layer = Conv2D(in_channels=1, out_channels=1, kernel_size=(2, 2), padding="VALID")
    params = {
        "w": jnp.ones((2, 2, 1, 1), dtype=jnp.float32),
        "b": jnp.array([0.5], dtype=jnp.float32),
    }
    inputs = jnp.arange(9, dtype=jnp.float32).reshape(1, 3, 3, 1)
    outputs = layer.apply(params, inputs)
    expected = jnp.array([[[[8.5], [12.5]], [[20.5], [24.5]]]], dtype=jnp.float32)
    assert outputs.shape == (1, 2, 2, 1)
    assert jnp.allclose(outputs, expected)


def test_conv2d_padding_and_stride_keeps_bias_only_output():
    layer = Conv2D(in_channels=2, out_channels=4, kernel_size=(3, 3), strides=(2, 2), padding="SAME")
    params = {
        "w": jnp.zeros((3, 3, 2, 4), dtype=jnp.float32),
        "b": jnp.full((4,), 0.7, dtype=jnp.float32),
    }
    inputs = jnp.ones((1, 5, 5, 2), dtype=jnp.float32)
    outputs = layer.apply(params, inputs)
    assert outputs.shape == (1, 3, 3, 4)
    assert jnp.allclose(outputs, 0.7)


def test_conv2d_backward_grads_exist():
    layer = Conv2D(in_channels=1, out_channels=1, kernel_size=(3, 3))
    params = layer.init(jax.random.PRNGKey(0))
    x = jnp.ones((2, 5, 5, 1))

    def loss_fn(wb):
        y = layer.apply(wb, x)
        return jnp.sum(y)

    grads = jax.grad(loss_fn)(params)
    assert grads["w"].shape == params["w"].shape
    assert grads["b"].shape == params["b"].shape


def test_batchnorm_shapes_and_normalization():
    layer = BatchNorm(num_features=3)
    params = layer.init(jax.random.PRNGKey(0))
    x = jnp.arange(4 * 2 * 2 * 3, dtype=jnp.float32).reshape(4, 2, 2, 3)
    y = layer.apply(params, x)
    assert y.shape == x.shape
    # Mean close to zero and variance close to one across batch+spatial
    y_mean = jnp.mean(y, axis=(0, 1, 2))
    y_var = jnp.var(y, axis=(0, 1, 2))
    assert jnp.allclose(y_mean, jnp.zeros_like(y_mean), atol=1e-5)
    assert jnp.allclose(y_var, jnp.ones_like(y_var), atol=1e-4)


def test_cross_entropy_logits_matches_manual():
    logits = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    loss_value = losses.cross_entropy_logits(logits, labels)
    softmax = jax.nn.softmax(logits, axis=-1)
    expected = -jnp.mean(jnp.log(jnp.array([softmax[0, 0], softmax[1, 1]])))
    assert jnp.allclose(loss_value, expected)


def test_cross_entropy_with_label_smoothing():
    logits = jnp.array([[2.0, -2.0]], dtype=jnp.float32)
    labels = jnp.array([0], dtype=jnp.int32)
    loss_value = losses.cross_entropy_logits(logits, labels, label_smoothing=0.2)
    log_probs = jax.nn.log_softmax(logits)
    smoothed = jnp.array([[0.9, 0.1]], dtype=jnp.float32)
    expected = -jnp.mean(jnp.sum(smoothed * log_probs, axis=-1))
    assert jnp.allclose(loss_value, expected)


def test_cross_entropy_invalid_smoothing_raises():
    logits = jnp.zeros((1, 2), dtype=jnp.float32)
    labels = jnp.zeros((1,), dtype=jnp.int32)
    with pytest.raises(ValueError):
        losses.cross_entropy_logits(logits, labels, label_smoothing=1.5)


def test_mse_basic():
    predictions = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    targets = jnp.array([1.0, 1.0, 2.0], dtype=jnp.float32)
    loss_value = losses.mse(predictions, targets)
    expected = jnp.mean(jnp.array([0.0, 1.0, 1.0], dtype=jnp.float32))
    assert jnp.allclose(loss_value, expected)


def test_nll_loss_matches_manual_and_reductions():
    log_probs = jnp.log(jnp.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=jnp.float32))
    targets = jnp.array([0, 2], dtype=jnp.int32)
    none = losses.nll_loss(log_probs, targets, reduction="none")
    expected_none = -jnp.log(jnp.array([0.7, 0.6], dtype=jnp.float32))
    assert jnp.allclose(none, expected_none)
    mean = losses.nll_loss(log_probs, targets, reduction="mean")
    assert jnp.allclose(mean, jnp.mean(expected_none))
    total = losses.nll_loss(log_probs, targets, reduction="sum")
    assert jnp.allclose(total, jnp.sum(expected_none))


def test_nll_loss_invalid_target_raises():
    log_probs = jnp.log(jnp.array([[0.5, 0.5]], dtype=jnp.float32))
    with pytest.raises(ValueError):
        losses.nll_loss(log_probs, jnp.array([2]))


def test_leaky_relu_behavior_and_slope():
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
    out_default = activations.leaky_relu(x)
    expected_default = jnp.array([-0.02, -0.01, 0.0, 1.0, 2.0], dtype=jnp.float32)
    assert jnp.allclose(out_default, expected_default)

    out_custom = activations.leaky_relu(x, negative_slope=0.2)
    expected_custom = jnp.array([-0.4, -0.2, 0.0, 1.0, 2.0], dtype=jnp.float32)
    assert jnp.allclose(out_custom, expected_custom)

    with pytest.raises(ValueError):
        _ = activations.leaky_relu(x, negative_slope=-0.1)
