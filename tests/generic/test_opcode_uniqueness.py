import re
import unittest
from pathlib import Path


class TestOpcodeUniqueness(unittest.TestCase):
    def test_no_duplicate_opcode_values(self):
        header = Path("include/generic/opcodes.cuh")
        self.assertTrue(header.exists())
        text = header.read_text()
        pattern = re.compile(r"constexpr\s+uint8_t\s+OP_[A-Z0-9_]+\s*=\s*0x([0-9A-Fa-f]{2});")
        values = [int(m.group(1), 16) for m in pattern.finditer(text)]
        self.assertEqual(len(values), len(set(values)), "Duplicate opcode values detected")


if __name__ == "__main__":
    unittest.main()

