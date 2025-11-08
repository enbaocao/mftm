import math
import unittest

from megakernels.generic_scheduler import ModelConfig, UniversalScheduler, Opcode


class TestSchedulerCounts(unittest.TestCase):
    def test_attention_and_residual_counts_per_layer_fused(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        attn_count = sum(1 for i in layer if i.opcode == Opcode.ATTENTION_PARTIAL)
        residual_count = sum(1 for i in layer if i.opcode == Opcode.RESIDUAL_ADD)
        self.assertEqual(attn_count, 1)
        self.assertEqual(residual_count, 2)

    def test_qkv_blocking_nonfused(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=False)
        # Collect MATMULs prior to the attention op
        qkv_matmuls = []
        for inst in layer:
            if inst.opcode == Opcode.ATTENTION_PARTIAL:
                break
            if inst.opcode == Opcode.MATMUL:
                qkv_matmuls.append(inst)
        self.assertGreater(len(qkv_matmuls), 0)
        total_n = sum(m.n_dim for m in qkv_matmuls)
        self.assertEqual(total_n, cfg.qkv_dim)
        # Check per-block sizes and block count
        self.assertTrue(all(m.n_dim <= cfg.matmul_block_n for m in qkv_matmuls))
        expected_blocks = math.ceil(cfg.qkv_dim / cfg.matmul_block_n)
        self.assertEqual(len(qkv_matmuls), expected_blocks)

    def test_lm_head_blocking(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        instrs = sch.build_full_model(use_fused_ops=True)
        # Find last norm index
        last_norm_idx = max(i for i, inst in enumerate(instrs) if inst.opcode in (Opcode.RMS_NORM, Opcode.LAYER_NORM))
        lm_head = instrs[last_norm_idx + 1 :]
        # Confirm block count equals ceil(vocab/block_n)
        expected_blocks = math.ceil(cfg.vocab_size / cfg.matmul_block_n)
        self.assertEqual(len(lm_head), expected_blocks)


if __name__ == "__main__":
    unittest.main()

