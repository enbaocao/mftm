import unittest

from megakernels.generic_scheduler import GenericInstruction, Opcode, ModelConfig


class TestInstructionCost(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig.from_llama_3_1b()

    def test_cost_matmul(self):
        inst = GenericInstruction(opcode=Opcode.MATMUL, m_dim=1, n_dim=8, k_dim=16)
        self.assertEqual(inst.cost(self.cfg), 1 * 8 * 16 * 2)

    def test_cost_norm(self):
        inst = GenericInstruction(opcode=Opcode.RMS_NORM, m_dim=1, n_dim=64)
        self.assertEqual(inst.cost(self.cfg), 1 * 64 * 3)

    def test_cost_attention(self):
        inst = GenericInstruction(opcode=Opcode.ATTENTION_PARTIAL, m_dim=4, n_dim=1, k_dim=32)
        self.assertEqual(inst.cost(self.cfg), 4 * 1 * 32 * 4)


if __name__ == "__main__":
    unittest.main()

