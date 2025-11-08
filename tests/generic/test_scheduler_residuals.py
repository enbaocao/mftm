import unittest

from megakernels.generic_scheduler import ModelConfig, UniversalScheduler, Opcode


class TestSchedulerResiduals(unittest.TestCase):
    def test_two_residual_adds_per_layer(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer_instrs = sch.build_transformer_layer(0, use_fused_ops=True)
        count = sum(1 for inst in layer_instrs if inst.opcode == Opcode.RESIDUAL_ADD)
        self.assertEqual(count, 2)

    def test_residual_add_order(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        layer_instrs = sch.build_transformer_layer(0, use_fused_ops=True)
        # First residual should follow a MATMUL (O projection)
        for i, inst in enumerate(layer_instrs):
            if inst.opcode == Opcode.RESIDUAL_ADD:
                self.assertGreater(i, 0)
                self.assertEqual(layer_instrs[i - 1].opcode, Opcode.MATMUL)
                break


if __name__ == "__main__":
    unittest.main()

