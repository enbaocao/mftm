import unittest

from megakernels.generic_scheduler import ModelConfig, UniversalScheduler, Opcode


class TestRopePresenceOrder(unittest.TestCase):
    def test_rope_before_attention_nonfused_llama(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=False)
        opcodes = [inst.opcode for inst in layer]
        self.assertIn(Opcode.ROPE_EMBED, opcodes)
        self.assertIn(Opcode.ATTENTION_PARTIAL, opcodes)
        self.assertLess(opcodes.index(Opcode.ROPE_EMBED), opcodes.index(Opcode.ATTENTION_PARTIAL))


if __name__ == "__main__":
    unittest.main()

