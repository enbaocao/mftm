import re
import unittest
from pathlib import Path

from megakernels.generic_scheduler import Opcode


def parse_opcodes_from_header(header_path):
    text = Path(header_path).read_text()
    pattern = re.compile(r"constexpr\s+uint8_t\s+(OP_[A-Z0-9_]+)\s*=\s*0x([0-9A-Fa-f]{2});")
    mapping = {}
    for m in pattern.finditer(text):
        name, hexval = m.group(1), m.group(2)
        mapping[name] = int(hexval, 16)
    return mapping


class TestOpcodeConsistency(unittest.TestCase):
    def test_core_opcodes_align(self):
        header = Path("include/generic/opcodes.cuh")
        self.assertTrue(header.exists(), "Missing include/generic/opcodes.cuh")
        mapping = parse_opcodes_from_header(header)

        # Spot-check a subset of important opcodes
        self.assertEqual(mapping["OP_MATMUL"], Opcode.MATMUL)
        self.assertEqual(mapping["OP_RMS_NORM"], Opcode.RMS_NORM)
        self.assertEqual(mapping["OP_LAYER_NORM"], Opcode.LAYER_NORM)
        self.assertEqual(mapping["OP_ATTENTION_PARTIAL"], Opcode.ATTENTION_PARTIAL)
        self.assertEqual(mapping["OP_ROPE_EMBED"], Opcode.ROPE_EMBED)
        self.assertEqual(mapping["OP_RESIDUAL_ADD"], Opcode.RESIDUAL_ADD)
        self.assertEqual(mapping["OP_FUSED_NORM_QKV_ROPE"], Opcode.FUSED_NORM_QKV_ROPE)
        self.assertEqual(mapping["OP_FUSED_NORM_MATMUL"], Opcode.FUSED_NORM_MATMUL)


if __name__ == "__main__":
    unittest.main()
