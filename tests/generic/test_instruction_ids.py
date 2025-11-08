import unittest

from megakernels.generic_scheduler import ModelConfig, UniversalScheduler


class TestInstructionIds(unittest.TestCase):
    def test_full_model_instruction_ids_sequential(self):
        cfg = ModelConfig.from_llama_3_1b()
        sch = UniversalScheduler(cfg)
        instrs = sch.build_full_model(use_fused_ops=True)
        ids = [i.instruction_id for i in instrs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sorted(ids), list(range(len(instrs))))


if __name__ == "__main__":
    unittest.main()

