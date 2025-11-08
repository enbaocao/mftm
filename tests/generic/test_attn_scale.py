import math
import unittest

from megakernels.generic_scheduler import ModelConfig, UniversalScheduler, Opcode


class TestAttentionScale(unittest.TestCase):
    def test_llama_attention_scale(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        attn = next(i for i in layer if i.opcode == Opcode.ATTENTION_PARTIAL)
        self.assertAlmostEqual(attn.scale_factor, 1.0 / math.sqrt(cfg.head_dim), places=6)

    def test_gpt2_attention_scale(self):
        cfg = ModelConfig.from_gpt2()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        attn = next(i for i in layer if i.opcode == Opcode.ATTENTION_PARTIAL)
        self.assertAlmostEqual(attn.scale_factor, 1.0 / math.sqrt(cfg.head_dim), places=6)


if __name__ == "__main__":
    unittest.main()

