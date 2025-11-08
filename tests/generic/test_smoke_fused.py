import os
import unittest
import importlib


class TestGenericSmokeFused(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_fused_norm_matmul_smoke(self):
        try:
            import torch
        except Exception:
            self.skipTest("PyTorch not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        try:
            mk_generic = importlib.import_module("mk_generic")
        except Exception as e:
            self.skipTest(f"mk_generic not importable: {e}")

        device = torch.device("cuda")

        # Dimensions: K=3, N=2
        K, N = 3, 2

        # Instruction: OP_FUSED_NORM_MATMUL (0xB0)
        inst = torch.zeros(32, dtype=torch.int32)
        inst[0] = (0xB0) | (0 << 8) | (0 << 16)
        inst[1] = (1) | (N << 16)  # m=1, n=N
        inst[2] = (K) | (0 << 16)  # k_dim=K
        inst[4] = 0  # input_offset_0 (x in a)
        inst[7] = 0  # output_offset (y in c)
        inst[5] = 0  # input_offset_1 (gamma in b)
        inst[8] = K  # weight_offset (W in b after gamma)
        eps = torch.tensor(0.0, dtype=torch.float32)
        inst[15] = eps.view(torch.int32)

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        # x, gamma, W
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        # Choose gamma so that norm(x)*gamma == x (i.e., gamma = rms)
        rms = torch.sqrt((x.pow(2).mean()))
        gamma = torch.tensor([rms.item(), rms.item(), rms.item()], device=device)
        W = torch.tensor([
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        ], device=device)

        a = x.clone()
        b = torch.cat([gamma, W])  # gamma then W rows
        c = torch.zeros(N, device=device)

        mk_generic.mk_generic_matmul(instructions, timings, a, b, c)

        # Expected: y = x @ W^T
        expected = torch.tensor([
            1.0,
            1.0 + 2.0 + 3.0,
        ], device=device)
        self.assertTrue(torch.allclose(c, expected, atol=1e-6))

    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_fused_norm_qkv_rope_reference(self):
        try:
            import torch
        except Exception:
            self.skipTest("PyTorch not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        try:
            mk_generic = importlib.import_module("mk_generic")
        except Exception as e:
            self.skipTest(f"mk_generic not importable: {e}")

        device = torch.device("cuda")

        # Dimensions
        K = 3
        N = 5  # qkv_dim for this synthetic test

        # Instruction: OP_FUSED_NORM_QKV_ROPE (0xB5)
        inst = torch.zeros(32, dtype=torch.int32)
        inst[0] = (0xB5)
        inst[1] = (1) | (N << 16)  # m=1, n=N
        inst[2] = (K)              # k_dim=K
        inst[4] = 0                # input_offset_0 (x in a)
        inst[5] = 0                # input_offset_1 (gamma in b)
        inst[8] = K                # weight_offset (W in b after gamma)
        inst[7] = 0                # output_offset (y in c)
        eps = torch.tensor(0.0, dtype=torch.float32)
        inst[15] = eps.view(torch.int32)  # scale_factor holds norm eps

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        # Data
        x = torch.tensor([1.0, 2.0, -1.0], device=device)
        gamma = torch.tensor([1.0, 0.5, 2.0], device=device)
        W = torch.tensor([
            # 5 rows of length 3
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
            2.0, -1.0, 0.0,
        ], device=device)

        a = x.clone()
        b = torch.cat([gamma, W])
        c = torch.zeros(N, device=device)

        mk_generic.mk_generic_matmul(instructions, timings, a, b, c)

        # Reference: y = (x / rms(x) * gamma) @ W^T; RoPE angle assumed 0 in smoke impl
        rms = torch.sqrt((x.pow(2).mean()))
        xn = x / rms * gamma
        Wm = W.view(N, K)
        expected = (Wm @ xn).contiguous()
        self.assertTrue(torch.allclose(c, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
