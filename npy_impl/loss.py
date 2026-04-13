"""
NumPy 版分类损失与基础指标。
"""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    softmax

    Args:
        logits: shape (N, C), dtype=float32/float64

    Returns:
        shape (N, C) 的概率分布
    """
    # softmax(x) = softmax(x - max(x)), 防止e^(logits)过大
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    交叉熵损失

    Args:
        logits: shape (N, C)
        labels: shape (N,)

    Returns:
        batch 平均 loss
    """
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - logsumexp
    losses = -log_probs[np.arange(logits.shape[0]), labels]
    return float(np.mean(losses))


def cross_entropy_backward(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    计算交叉熵损失对 logits 的梯度。

    Args:
        logits: shape (N, C)
        labels: shape (N,)

    Returns:
        dL/dlogits, shape (N, C)
    """
    probs = softmax(logits).astype(np.float32, copy=False)
    grad_logits = probs.copy()
    grad_logits[np.arange(logits.shape[0]), labels] -= 1.0
    grad_logits /= logits.shape[0]
    return grad_logits.astype(np.float32, copy=False)


def top1_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    计算 top1 准确率，返回百分比。
    """
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == labels) * 100.0)
