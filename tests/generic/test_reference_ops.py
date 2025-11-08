import math
import unittest


def rms_norm(x, gamma=None, eps=1e-5):
    n = len(x)
    sum_sq = sum(v * v for v in x)
    rms = math.sqrt(sum_sq / n + eps)
    inv = 1.0 / rms
    if gamma is None:
        gamma = [1.0] * n
    return [(x[i] * inv) * gamma[i] for i in range(n)]


def matmul_vec(W_rows, x):
    # W_rows: list of rows (N x K), x: length K
    out = []
    for row in W_rows:
        out.append(sum(w * xi for w, xi in zip(row, x)))
    return out


def fused_norm_matmul(x, gamma, W_rows, eps=1e-5):
    xh = rms_norm(x, gamma, eps)
    return matmul_vec(W_rows, xh)


def rope_rotate(x, angle):
    # Pairwise rotation across even-odd dims
    out = list(x)
    c, s = math.cos(angle), math.sin(angle)
    for d in range(0, len(x), 2):
        x0 = x[d]
        x1 = x[d + 1] if d + 1 < len(x) else 0.0
        out[d] = x0 * c - x1 * s
        if d + 1 < len(x):
            out[d + 1] = x0 * s + x1 * c
    return out


def attention_decode(q, K_rows, V_rows, scale=1.0):
    logits = [sum(qi * kij for qi, kij in zip(q, k)) * scale for k in K_rows]
    max_logit = max(logits) if logits else 0.0
    exps = [math.exp(l - max_logit) for l in logits]
    denom = sum(exps) if exps else 1.0
    weights = [e / denom for e in exps]
    # Weighted sum of V rows
    out = [0.0] * len(V_rows[0])
    for w, v in zip(weights, V_rows):
        for i, vi in enumerate(v):
            out[i] += w * vi
    return out


class TestReferenceOps(unittest.TestCase):
    def test_fused_norm_matmul_reference(self):
        x = [1.0, 2.0, -1.0]
        gamma = [1.0, 0.5, 2.0]
        W = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        y = fused_norm_matmul(x, gamma, W, eps=0.0)
        # Manual reference
        n = len(x)
        rms = math.sqrt(sum(v * v for v in x) / n)
        xn = [(x[i] / rms) * gamma[i] for i in range(n)]
        expected = matmul_vec(W, xn)
        self.assertEqual(len(y), len(expected))
        for a, b in zip(y, expected):
            self.assertAlmostEqual(a, b, places=7)

    def test_rope_rotate(self):
        x = [2.0, 3.0]
        y = rope_rotate(x, math.pi / 2.0)
        self.assertAlmostEqual(y[0], -3.0, places=7)
        self.assertAlmostEqual(y[1], 2.0, places=7)

    def test_rope_zero_angle_odd_dim(self):
        x = [2.0, 3.0, -4.0]
        y = rope_rotate(x, 0.0)
        self.assertEqual(y, x)

    def test_attention_decode_small(self):
        q = [1.0, 0.0, 0.0]
        K = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        V = [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
        ]
        y = attention_decode(q, K, V, scale=1.0)
        # Softmax([1,0])
        import math as _m

        w0 = 1.0 / (1.0 + _m.exp(-1.0))
        w1 = 1.0 - w0
        expected = [w0 * 10.0 + w1 * 0.0, w1 * 20.0, 0.0]
        for a, b in zip(y, expected):
            self.assertAlmostEqual(a, b, places=6)


def attention_partial_with_lse(q, K_rows, V_rows, scale=1.0):
    """
    Compute partial attention with LSE (log-sum-exp) values for reduction.
    Returns: (output, max_logit, log_sum_exp)
    """
    logits = [sum(qi * kij for qi, kij in zip(q, k)) * scale for k in K_rows]
    max_logit = max(logits) if logits else 0.0
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exps) if exps else 1.0
    weights = [e / sum_exp for e in exps]

    # Weighted sum of V rows
    out = [0.0] * len(V_rows[0])
    for w, v in zip(weights, V_rows):
        for i, vi in enumerate(v):
            out[i] += w * vi

    return out, max_logit, math.log(sum_exp)


def attention_reduce(partial_outputs, lse_values):
    """
    Combine multiple attention partials using LSE trick.
    partial_outputs: list of output vectors from each partial
    lse_values: list of (max_logit, log_sum_exp) tuples
    Returns: final attention output
    """
    if not partial_outputs:
        return []

    head_dim = len(partial_outputs[0])
    num_partials = len(partial_outputs)

    # Step 1: Find global max
    global_max = max(lse[0] for lse in lse_values)

    # Step 2: Compute corrected sum and output
    corrected_sum = 0.0
    final_output = [0.0] * head_dim

    for p in range(num_partials):
        m_i, log_s_i = lse_values[p]
        correction = math.exp(m_i - global_max + log_s_i)
        corrected_sum += correction

        for d in range(head_dim):
            final_output[d] += partial_outputs[p][d] * correction

    # Step 3: Normalize
    inv_sum = 1.0 / corrected_sum
    for d in range(head_dim):
        final_output[d] *= inv_sum

    return final_output


class TestAttentionPrefill(unittest.TestCase):
    def test_attention_partial_with_lse(self):
        """Test partial attention with LSE output"""
        q = [1.0, 0.0, 0.0]
        K = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        V = [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
        ]

        o, max_logit, log_sum_exp = attention_partial_with_lse(q, K, V, scale=1.0)

        # Check max_logit (should be 1.0 from first logit)
        self.assertAlmostEqual(max_logit, 1.0, places=6)

        # Check that output matches regular attention
        o_regular = attention_decode(q, K, V, scale=1.0)
        for a, b in zip(o, o_regular):
            self.assertAlmostEqual(a, b, places=6)

    def test_attention_reduce_single_partial(self):
        """Test reduction with single partial (should be identity)"""
        partial_output = [1.0, 2.0, 3.0]
        lse_values = [(0.5, math.log(2.0))]

        result = attention_reduce([partial_output], lse_values)

        # Should equal the input when there's only one partial
        for a, b in zip(result, partial_output):
            self.assertAlmostEqual(a, b, places=6)

    def test_attention_reduce_two_partials(self):
        """Test reduction with two partials"""
        # Simulate two KV blocks
        q = [1.0, 0.0, 0.0]

        # First partial: first 2 KV pairs
        K1 = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
        V1 = [[10.0, 0.0, 0.0], [5.0, 5.0, 0.0]]
        o1, m1, log_s1 = attention_partial_with_lse(q, K1, V1, scale=1.0)

        # Second partial: next 2 KV pairs
        K2 = [[0.0, 1.0, 0.0], [-0.5, 0.5, 0.0]]
        V2 = [[0.0, 20.0, 0.0], [-5.0, 5.0, 0.0]]
        o2, m2, log_s2 = attention_partial_with_lse(q, K2, V2, scale=1.0)

        # Reduce
        reduced = attention_reduce([o1, o2], [(m1, log_s1), (m2, log_s2)])

        # Compare with full attention
        K_full = K1 + K2
        V_full = V1 + V2
        expected = attention_decode(q, K_full, V_full, scale=1.0)

        for a, b in zip(reduced, expected):
            self.assertAlmostEqual(a, b, places=5)

    def test_attention_reduce_three_partials(self):
        """Test reduction with three partials (more realistic prefill)"""
        q = [1.0, 0.5, 0.0]

        # Three KV blocks
        K1 = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]
        V1 = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]
        o1, m1, log_s1 = attention_partial_with_lse(q, K1, V1, scale=1.0)

        K2 = [[0.0, 1.0, 0.0], [0.1, 0.9, 0.0]]
        V2 = [[0.0, 2.0, 0.0], [0.2, 1.8, 0.0]]
        o2, m2, log_s2 = attention_partial_with_lse(q, K2, V2, scale=1.0)

        K3 = [[0.5, 0.5, 0.0], [0.3, 0.7, 0.0]]
        V3 = [[0.5, 1.5, 0.0], [0.3, 1.7, 0.0]]
        o3, m3, log_s3 = attention_partial_with_lse(q, K3, V3, scale=1.0)

        # Reduce
        reduced = attention_reduce([o1, o2, o3], [(m1, log_s1), (m2, log_s2), (m3, log_s3)])

        # Compare with full attention
        K_full = K1 + K2 + K3
        V_full = V1 + V2 + V3
        expected = attention_decode(q, K_full, V_full, scale=1.0)

        for a, b in zip(reduced, expected):
            self.assertAlmostEqual(a, b, places=5)


if __name__ == "__main__":
    unittest.main()

