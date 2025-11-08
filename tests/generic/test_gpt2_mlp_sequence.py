import unittest

from megakernels.generic_scheduler import ModelConfig, UniversalScheduler, Opcode


class TestGpt2MlpSequence(unittest.TestCase):
    def test_mlp_ops_sequence(self):
        cfg = ModelConfig.from_gpt2()
        sch = UniversalScheduler(cfg)
        layer = sch.build_transformer_layer(0, use_fused_ops=True)
        # Find a contiguous subsequence [MATMUL, GELU, MATMUL, RESIDUAL_ADD]
        ops = [inst.opcode for inst in layer]
        pattern = [Opcode.MATMUL, Opcode.GELU, Opcode.MATMUL, Opcode.RESIDUAL_ADD]
        found = False
        for i in range(len(ops) - 3):
            if ops[i : i + 4] == pattern:
                found = True
                break
        self.assertTrue(found, "Expected GPT-2 MLP sequence not found")


if __name__ == "__main__":
    unittest.main()

