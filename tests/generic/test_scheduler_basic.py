import os
import unittest

from megakernels.generic_scheduler import (
    ModelConfig,
    UniversalScheduler,
    Opcode,
    NormType,
)


class TestGenericSchedulerBasic(unittest.TestCase):
    def test_llama_layer_instructions(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer_instrs = sch.build_transformer_layer(layer_idx=0, use_fused_ops=True)

        self.assertGreaterEqual(len(layer_instrs), 5)

        # First instruction should be fused norm+QKV(+rope)
        self.assertEqual(layer_instrs[0].opcode, Opcode.FUSED_NORM_QKV_ROPE)

        # Attention should be present
        opcodes = [inst.opcode for inst in layer_instrs]
        self.assertIn(Opcode.ATTENTION_PARTIAL, opcodes)

        # MLP residual add should be present
        self.assertIn(Opcode.RESIDUAL_ADD, opcodes)

    def test_gpt2_layer_instructions(self):
        cfg = ModelConfig.from_gpt2()
        sch = UniversalScheduler(cfg)
        layer_instrs = sch.build_transformer_layer(layer_idx=0, use_fused_ops=True)

        self.assertGreaterEqual(len(layer_instrs), 5)

        # GPT-2 uses LayerNorm
        has_layer_norm = any(inst.opcode == Opcode.LAYER_NORM for inst in layer_instrs)
        # With fused ops, explicit LAYER_NORM may not appear; allow either
        fused_first = layer_instrs[0].opcode == Opcode.FUSED_NORM_QKV_ROPE
        self.assertTrue(has_layer_norm or fused_first)

        # No RoPE op in GPT-2 sequence (no explicit ROPE_EMBED expected)
        self.assertNotIn(Opcode.ROPE_EMBED, [inst.opcode for inst in layer_instrs])

    def test_non_fused_path_contains_norm_and_rope(self):
        # Force non-fused to check explicit ops presence
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer_instrs = sch.build_transformer_layer(layer_idx=0, use_fused_ops=False)
        opcodes = [inst.opcode for inst in layer_instrs]
        self.assertIn(Opcode.RMS_NORM, opcodes)
        self.assertIn(Opcode.MATMUL, opcodes)
        # Llama uses RoPE
        self.assertIn(Opcode.ROPE_EMBED, opcodes)

    def test_full_model_sizes(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        instrs = sch.build_full_model(use_fused_ops=True)
        # Expect at least 5 instructions per layer plus final head
        self.assertGreaterEqual(len(instrs), cfg.num_layers * 5)

    def test_mistral_layer_costs(self):
        cfg = ModelConfig.from_mistral_7b()
        sch = UniversalScheduler(cfg)
        layer_instrs = sch.build_transformer_layer(layer_idx=0, use_fused_ops=True)
        total_cost = sum(inst.cost(cfg) for inst in layer_instrs)
        # Sanity: non-zero and positive
        self.assertGreater(total_cost, 0)


if __name__ == "__main__":
    unittest.main()
