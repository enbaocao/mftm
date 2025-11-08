import os
import unittest
import importlib


class TestGenericSmokeDemo(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_matmul_smoke(self):
        # Optional CUDA smoke test; requires compiled extension and torch CUDA
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

        # Build a single MATMUL instruction (packed 32-int format)
        # opcode=0x30 (MATMUL), flags=0, layer_idx=0
        # m=1, n=4, k=3; offsets: a@0, b@0, c@0
        inst = torch.zeros(32, dtype=torch.int32)
        inst[0] = (0x30) | (0 << 8) | (0 << 16)
        inst[1] = (1) | (4 << 16)
        inst[2] = (3) | (0 << 16)
        inst[3] = (0) | (0 << 16)
        inst[4] = 0  # input_offset_0
        inst[8] = 0  # weight_offset
        inst[7] = 0  # output_offset

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        # Data: a = [1,2,3], b rows = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        b = torch.tensor([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ], dtype=torch.float32, device=device)
        c = torch.zeros(4, dtype=torch.float32, device=device)
        barriers = torch.zeros(256, dtype=torch.int32, device=device)

        mk_generic.mk_generic_matmul(instructions, timings, a, b, c, barriers)

        expected = torch.tensor([1.0, 2.0, 3.0, 6.0], device=device)
        self.assertTrue(torch.allclose(c, expected, atol=1e-6))

    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_rmsnorm_smoke(self):
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

        # Build a single RMS_NORM instruction
        # opcode=0x50, eps in scale_factor slot (buf[15])
        n = 4
        inst = torch.zeros(32, dtype=torch.int32)
        inst[0] = (0x50) | (0 << 8) | (0 << 16)
        inst[1] = (1) | (n << 16)  # m=1, n=n
        inst[7] = 0  # output_offset
        inst[4] = 0  # input_offset_0
        inst[8] = n  # weight_offset placed after input; we'll arrange buffers accordingly
        # write scale_factor as float into buf[15]
        eps = torch.tensor(1e-5, dtype=torch.float32)
        inst[15] = eps.view(torch.int32)

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        # Prepare input x and weight gamma of length n
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device)
        gamma = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)

        # The binding expects three buffers a,b,c; we pack x||gamma into a and b:
        # a: input x starting at 0; b: gamma starting at offset n
        a = x.clone()
        b = torch.cat([torch.zeros(n, device=device, dtype=torch.float32), gamma])
        c = torch.zeros(n, dtype=torch.float32, device=device)
        barriers = torch.zeros(256, dtype=torch.int32, device=device)

        mk_generic.mk_generic_matmul(instructions, timings, a, b, c, barriers)

        # Compute expected RMSNorm(x) with gamma=1
        rms = torch.sqrt((x.pow(2).mean()) + 1e-5)
        expected = x / rms
        self.assertTrue(torch.allclose(c, expected, atol=1e-5))

    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_layernorm_smoke(self):
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

        # Build a single LAYER_NORM instruction: opcode=0x51
        n = 4
        inst = torch.zeros(32, dtype=torch.int32)
        # Keep higher 16 bits within signed int32 range (<= 0x7FFF)
        inst[0] = (0x51) | (0xAA << 8) | (0x7A7A << 16)
        inst[1] = (1) | (n << 16)  # m=1, n=n
        inst[7] = 0  # output_offset
        inst[4] = 0  # input_offset_0
        inst[8] = n  # weight_offset
        eps = torch.tensor(1e-5, dtype=torch.float32)
        inst[15] = eps.view(torch.int32)

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device)
        gamma = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        a = x.clone()
        b = torch.cat([torch.zeros(n, device=device, dtype=torch.float32), gamma])
        c = torch.zeros(n, dtype=torch.float32, device=device)
        barriers = torch.zeros(256, dtype=torch.int32, device=device)

        mk_generic.mk_generic_matmul(instructions, timings, a, b, c, barriers)

        mean = x.mean()
        var = ((x - mean) ** 2).mean()
        expected = (x - mean) / torch.sqrt(var + 1e-5)
        self.assertTrue(torch.allclose(c, expected, atol=1e-5))

    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_matmul_dispatch_masking(self):
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

        # Same as matmul smoke, but set non-zero flags/layer bits in word 0
        inst = torch.zeros(32, dtype=torch.int32)
        # Keep higher 16 bits within signed int32 range
        inst[0] = (0x30) | (0x5A << 8) | (0x7AFE << 16)
        inst[1] = (1) | (4 << 16)
        inst[2] = (3) | (0 << 16)
        inst[3] = (0) | (0 << 16)
        inst[4] = 0
        inst[8] = 0
        inst[7] = 0

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        b = torch.tensor([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ], dtype=torch.float32, device=device)
        c = torch.zeros(4, dtype=torch.float32, device=device)
        barriers = torch.zeros(256, dtype=torch.int32, device=device)

        mk_generic.mk_generic_matmul(instructions, timings, a, b, c, barriers)

        expected = torch.tensor([1.0, 2.0, 3.0, 6.0], device=device)
        self.assertTrue(torch.allclose(c, expected, atol=1e-6))

    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_attention_partial_smoke(self):
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

        # Build a single ATTENTION_PARTIAL instruction: opcode=0x70
        head_dim = 3
        kv_len = 2
        inst = torch.zeros(32, dtype=torch.int32)
        inst[0] = (0x70) | (0 << 8) | (0 << 16)
        inst[1] = (1) | (1 << 16)  # m=1 head, n=1 token
        inst[2] = (head_dim) | (0 << 16)  # k_dim=head_dim
        inst[10] = (0) | (kv_len << 16)  # reduction_factor in high 16 bits
        inst[4] = 0  # q offset in a
        inst[5] = 0  # k offset in b
        inst[6] = head_dim * kv_len  # v offset in b after K
        inst[7] = 0  # output offset in c
        # scale in buf[15]
        scale = torch.tensor(1.0, dtype=torch.float32)
        inst[15] = scale.view(torch.int32)

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        # q=[1,0,0]; K0=[1,0,0], K1=[0,1,0]; V0=[10,0,0], V1=[0,20,0]
        q = torch.tensor([1.0, 0.0, 0.0], device=device)
        K = torch.tensor([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ], device=device)
        V = torch.tensor([
            10.0, 0.0, 0.0,
            0.0, 20.0, 0.0,
        ], device=device)
        a = q.clone()
        b = torch.cat([K, V])
        c = torch.zeros(head_dim, device=device)
        barriers = torch.zeros(256, dtype=torch.int32, device=device)

        mk_generic.mk_generic_matmul(instructions, timings, a, b, c, barriers)

        logits = torch.tensor([1.0, 0.0], device=device)
        w = torch.softmax(logits, dim=0)
        expected = w[0] * V[:head_dim] + w[1] * V[head_dim:]
        self.assertTrue(torch.allclose(c, expected, atol=1e-5))

    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_attention_single_kv(self):
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

        head_dim = 3
        kv_len = 1
        inst = torch.zeros(32, dtype=torch.int32)
        inst[0] = (0x70)
        inst[2] = (head_dim)
        inst[10] = (0) | (kv_len << 16)
        inst[4] = 0  # q
        inst[5] = 0  # k
        inst[6] = head_dim * kv_len  # v
        inst[7] = 0
        inst[15] = torch.tensor(1.0, dtype=torch.float32).view(torch.int32)

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        q = torch.tensor([1.0, 2.0, 3.0], device=device)
        K = q.clone()  # exact match so weight ~ 1
        V0 = torch.tensor([0.5, -1.0, 2.0], device=device)
        a = q.clone()
        b = torch.cat([K, V0])
        c = torch.zeros(head_dim, device=device)
        barriers = torch.zeros(256, dtype=torch.int32, device=device)

        mk_generic.mk_generic_matmul(instructions, timings, a, b, c, barriers)

        # With a single KV entry, the softmax weight is 1.0 -> output = V0
        self.assertTrue(torch.allclose(c, V0, atol=1e-6))

    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_rope_embed_smoke(self):
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

        # opcode=0x72, head_dim=2, angle=pi/2
        inst = torch.zeros(32, dtype=torch.int32)
        inst[0] = (0x72) | (0 << 8) | (0 << 16)
        inst[2] = (2) | (0 << 16)  # k_dim=2
        inst[4] = 0  # input_offset_0
        inst[7] = 0  # output_offset
        angle = torch.tensor(3.14159265 / 2.0, dtype=torch.float32)
        inst[15] = angle.view(torch.int32)

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        x = torch.tensor([2.0, 3.0], device=device)
        a = x.clone()
        b = torch.zeros(1, device=device)  # unused
        c = torch.zeros(2, device=device)
        barriers = torch.zeros(256, dtype=torch.int32, device=device)

        mk_generic.mk_generic_matmul(instructions, timings, a, b, c, barriers)

        # Rotate by 90 degrees: [x,y] -> [-y, x]
        expected = torch.tensor([-3.0, 2.0], device=device)
        self.assertTrue(torch.allclose(c, expected, atol=1e-5))

    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_rope_angle_zero_odd_dim(self):
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

        # head_dim=3, angle=0 => y = x
        inst = torch.zeros(32, dtype=torch.int32)
        inst[0] = (0x72)
        inst[2] = (3)
        inst[4] = 0
        inst[7] = 0
        inst[15] = torch.tensor(0.0, dtype=torch.float32).view(torch.int32)

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        x = torch.tensor([2.0, 3.0, -4.0], device=device)
        a = x.clone()
        b = torch.zeros(1, device=device)
        c = torch.zeros(3, device=device)
        barriers = torch.zeros(256, dtype=torch.int32, device=device)

        mk_generic.mk_generic_matmul(instructions, timings, a, b, c, barriers)
        self.assertTrue(torch.allclose(c, x, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
