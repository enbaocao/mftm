import unittest

from megakernels.generic_scheduler import ModelConfig


class TestModelConfig(unittest.TestCase):
    def test_llama_dimensions(self):
        cfg = ModelConfig.from_llama_3_1b()
        self.assertEqual(cfg.q_dim, 32 * 64)
        self.assertEqual(cfg.kv_dim, 8 * 64)
        self.assertEqual(cfg.qkv_dim, 32 * 64 + 2 * 8 * 64)

    def test_mistral_dimensions(self):
        cfg = ModelConfig.from_mistral_7b()
        self.assertEqual(cfg.kv_dim, cfg.num_kv_heads * cfg.head_dim)
        self.assertTrue(cfg.has_sliding_window)
        self.assertGreater(cfg.sliding_window_size, 0)


if __name__ == "__main__":
    unittest.main()

