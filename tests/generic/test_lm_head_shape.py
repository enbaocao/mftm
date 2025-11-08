import unittest

from megakernels.generic_scheduler import ModelConfig, UniversalScheduler, Opcode


class TestLmHeadShape(unittest.TestCase):
    def test_lm_head_blocks_cover_vocab(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        instrs = sch.build_full_model(use_fused_ops=True)

        # Find final norm index
        last_norm_idx = max(i for i, inst in enumerate(instrs) if inst.opcode in (Opcode.RMS_NORM, Opcode.LAYER_NORM))
        lm_head = instrs[last_norm_idx + 1 :]

        # LM head should be a non-empty sequence of MATMULs whose total n_dim equals vocab_size
        self.assertGreater(len(lm_head), 0)
        self.assertTrue(all(inst.opcode == Opcode.MATMUL for inst in lm_head))

        total_n = sum(inst.n_dim for inst in lm_head)
        self.assertEqual(total_n, cfg.vocab_size)
        # Each block should be <= matmul block size
        self.assertTrue(all(inst.n_dim <= cfg.matmul_block_n for inst in lm_head))


if __name__ == "__main__":
    unittest.main()

