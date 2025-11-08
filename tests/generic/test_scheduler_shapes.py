import unittest

from megakernels.generic_scheduler import ModelConfig, UniversalScheduler, Opcode


class TestSchedulerShapes(unittest.TestCase):
    def test_fused_qkv_dims(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        fused = layer[0]
        self.assertEqual(fused.opcode, Opcode.FUSED_NORM_QKV_ROPE)
        self.assertEqual(fused.m_dim, 1)
        self.assertEqual(fused.n_dim, cfg.qkv_dim)
        self.assertEqual(fused.k_dim, cfg.hidden_dim)

    def test_attention_dims(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        attn = [i for i in layer if i.opcode == Opcode.ATTENTION_PARTIAL][0]
        self.assertEqual(attn.m_dim, cfg.num_q_heads)
        self.assertEqual(attn.k_dim, cfg.head_dim)
        self.assertEqual(attn.n_dim, 1)

    def test_non_fused_rope_dims(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=False)
        rope = [i for i in layer if i.opcode == Opcode.ROPE_EMBED][0]
        self.assertEqual(rope.m_dim, cfg.num_q_heads)
        self.assertEqual(rope.k_dim, cfg.head_dim)


if __name__ == "__main__":
    unittest.main()

