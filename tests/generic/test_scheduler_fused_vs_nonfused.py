import unittest

from megakernels.generic_scheduler import ModelConfig, UniversalScheduler, Opcode


class TestSchedulerFusedVsNonFused(unittest.TestCase):
    def test_fused_first_instr_fields(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        first = layer[0]
        self.assertEqual(first.opcode, Opcode.FUSED_NORM_QKV_ROPE)
        self.assertEqual(first.n_dim, cfg.qkv_dim)
        self.assertEqual(first.k_dim, cfg.hidden_dim)
        # scale_factor is used to carry norm eps in our design
        self.assertAlmostEqual(first.scale_factor, cfg.norm_eps, places=6)

    def test_fused_vs_nonfused_count(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        fused = sch.build_transformer_layer(0, use_fused_ops=True)
        non_fused = sch.build_transformer_layer(0, use_fused_ops=False)
        self.assertLessEqual(len(fused), len(non_fused))

    def test_o_projection_structure(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        # Find first RESIDUAL_ADD and assert previous is MATMUL
        idx = next(i for i, inst in enumerate(layer) if inst.opcode == Opcode.RESIDUAL_ADD)
        self.assertGreater(idx, 0)
        self.assertEqual(layer[idx - 1].opcode, Opcode.MATMUL)

    def test_final_norm_layer_idx(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        full = sch.build_full_model(use_fused_ops=True)
        # final norm is appended before LM head
        # Search backwards for first norm
        norm = next(inst for inst in reversed(full) if inst.opcode in (Opcode.RMS_NORM, Opcode.LAYER_NORM))
        self.assertEqual(norm.layer_idx, cfg.num_layers)

    def test_attention_scale_factor(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        attn = next(inst for inst in layer if inst.opcode == Opcode.ATTENTION_PARTIAL)
        # default attn scale is 1/sqrt(head_dim)
        expected = 1.0 / (cfg.head_dim ** 0.5)
        self.assertAlmostEqual(attn.scale_factor, expected, places=6)


if __name__ == "__main__":
    unittest.main()

