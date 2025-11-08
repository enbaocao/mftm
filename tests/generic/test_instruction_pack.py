import unittest
import numpy as np

from megakernels.generic_scheduler import GenericInstruction, Opcode


class TestInstructionPack(unittest.TestCase):
    def test_serialize_length_and_types(self):
        inst = GenericInstruction(
            opcode=Opcode.MATMUL,
            flags=0xA5,
            layer_idx=3,
            m_dim=1,
            n_dim=128,
            k_dim=256,
            block_idx_m=0,
            block_idx_n=1,
            block_idx_k=2,
            input_offset_0=123,
            input_offset_1=456,
            input_offset_2=789,
            output_offset=321,
            weight_offset=654,
            scratch_offset=987,
            head_config=0x1234,
            reduction_factor=7,
            seq_pos=5,
            batch_idx=0,
            dependency_mask=0x55AA,
            sync_slot=3,
            sync_count=1,
            parent_instr_id=42,
            instruction_id=99,
            scale_factor=0.125,
        )

        buf = inst.serialize()
        self.assertEqual(buf.shape, (32,))
        self.assertEqual(buf.dtype, np.int32)

        # Check bit packing of first word (opcode|flags|layer)
        w0 = int(buf[0])
        self.assertEqual(w0 & 0xFF, Opcode.MATMUL)
        self.assertEqual((w0 >> 8) & 0xFF, 0xA5)
        self.assertEqual((w0 >> 16) & 0xFFFF, 3)

        # Dimensions
        self.assertEqual(int(buf[1]) & 0xFFFF, 1)
        self.assertEqual((int(buf[1]) >> 16) & 0xFFFF, 128)
        self.assertEqual(int(buf[2]) & 0xFFFF, 256)
        self.assertEqual((int(buf[2]) >> 16) & 0xFFFF, 0)
        self.assertEqual(int(buf[3]) & 0xFFFF, 1)
        self.assertEqual((int(buf[3]) >> 16) & 0xFFFF, 2)

        # Offsets
        self.assertEqual(int(buf[4]), 123)
        self.assertEqual(int(buf[5]), 456)
        self.assertEqual(int(buf[6]), 789)
        self.assertEqual(int(buf[7]), 321)
        self.assertEqual(int(buf[8]), 654)
        self.assertEqual(int(buf[9]), 987)

        # Config and sync
        self.assertEqual(int(buf[10]) & 0xFFFF, 0x1234)
        self.assertEqual((int(buf[10]) >> 16) & 0xFFFF, 7)
        self.assertEqual(int(buf[11]) & 0xFFFF, 5)
        self.assertEqual((int(buf[11]) >> 16) & 0xFFFF, 0)
        self.assertEqual(int(buf[12]) & 0xFFFF, 0x55AA)
        self.assertEqual((int(buf[12]) >> 16) & 0xFF, 3)
        self.assertEqual((int(buf[12]) >> 24) & 0xFF, 1)
        self.assertEqual(int(buf[13]), 42)

        # Metadata
        self.assertEqual(int(buf[14]), 99)
        self.assertEqual(np.float32(inst.scale_factor).view(np.int32), buf[15])

        # Padding (remaining slots) should be zero
        self.assertTrue(np.all(buf[16:] == 0))


if __name__ == "__main__":
    unittest.main()
