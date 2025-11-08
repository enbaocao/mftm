import unittest

from megakernels.generic_scheduler import ModelConfig, UniversalScheduler, Opcode, AttentionType


class TestAttentionHeadConfig(unittest.TestCase):
    def test_llama_head_config_gqa(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        attn = next(inst for inst in layer if inst.opcode == Opcode.ATTENTION_PARTIAL)
        self.assertEqual(attn.head_config, AttentionType.GQA)

    def test_gpt2_head_config_mha(self):
        cfg = ModelConfig.from_gpt2()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        attn = next(inst for inst in layer if inst.opcode == Opcode.ATTENTION_PARTIAL)
        self.assertEqual(attn.head_config, AttentionType.MHA)


if __name__ == "__main__":
    unittest.main()

