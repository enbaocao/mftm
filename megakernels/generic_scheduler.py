"""
Generic Instruction Scheduler for Multi-Model Support

This module provides a model-agnostic scheduler that generates instruction
sequences for different transformer architectures (Llama, GPT, Mistral, etc.)
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple
import numpy as np


# ============================================================================
# Opcode Definitions (matching C++ opcodes.cuh)
# ============================================================================

class Opcode(IntEnum):
    # Control
    NOOP = 0x00
    BARRIER = 0x01

    # Memory
    LOAD_ACTIVATION = 0x10
    STORE_ACTIVATION = 0x11
    LOAD_WEIGHT_TILE = 0x12
    CACHE_APPEND = 0x14

    # Linear algebra
    MATMUL = 0x30
    MATVEC = 0x31

    # Normalization
    RMS_NORM = 0x50
    LAYER_NORM = 0x51

    # Attention
    ATTENTION_PARTIAL = 0x70
    ATTENTION_REDUCE = 0x71
    ROPE_EMBED = 0x72

    # Activations
    SILU = 0x90
    GELU = 0x91
    SWIGLU = 0x93

    # Element-wise
    ADD = 0xA0
    RESIDUAL_ADD = 0xA2

    # Fused operations
    FUSED_NORM_MATMUL = 0xB0
    FUSED_ROPE_APPEND = 0xB1
    FUSED_GATE_ACT = 0xB2
    FUSED_QKV_PROJ = 0xB3
    FUSED_NORM_QKV_ROPE = 0xB5


class AttentionType(IntEnum):
    MHA = 0x0000
    GQA = 0x0001
    MQA = 0x0002
    MLA = 0x0003


class NormType(IntEnum):
    NONE = 0
    RMS_NORM = 1
    LAYER_NORM = 2


class ActivationType(IntEnum):
    NONE = 0
    RELU = 1
    GELU = 2
    SILU = 3
    SWIGLU = 4
    GEGLU = 5


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Runtime model configuration matching C++ RuntimeModelConfig"""

    # Architecture
    num_layers: int
    hidden_dim: int
    intermediate_dim: int
    vocab_size: int

    # Attention
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int

    # Attention pattern
    attention_type: AttentionType
    has_rope: bool = True
    has_sliding_window: bool = False
    sliding_window_size: int = 0

    rope_theta: float = 10000.0
    rope_scaling: float = 1.0
    attn_scale: Optional[float] = None

    # MLP
    mlp_activation: ActivationType = ActivationType.SWIGLU
    mlp_gated: bool = True

    # Normalization
    norm_type: NormType = NormType.RMS_NORM
    norm_eps: float = 1e-5

    # Tiling
    matmul_block_m: int = 16
    matmul_block_n: int = 16
    matmul_block_k: int = 512
    kv_block_size: int = 16

    # Hardware
    sm_count: int = 132
    max_batch_size: int = 1

    def __post_init__(self):
        if self.attn_scale is None:
            self.attn_scale = 1.0 / np.sqrt(self.head_dim)

    @property
    def qkv_dim(self) -> int:
        return self.num_q_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @classmethod
    def from_llama_3_1b(cls, sm_count: int = 132) -> "ModelConfig":
        """Create config for Llama 3.2 1B"""
        return cls(
            num_layers=16,
            hidden_dim=2048,
            intermediate_dim=8192,
            vocab_size=128256,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=64,
            max_seq_len=8192,
            attention_type=AttentionType.GQA,
            rope_theta=500000.0,
            sm_count=sm_count,
        )

    @classmethod
    def from_gpt2(cls, sm_count: int = 132) -> "ModelConfig":
        """Create config for GPT-2 124M"""
        return cls(
            num_layers=12,
            hidden_dim=768,
            intermediate_dim=3072,
            vocab_size=50257,
            num_q_heads=12,
            num_kv_heads=12,
            head_dim=64,
            max_seq_len=1024,
            attention_type=AttentionType.MHA,
            has_rope=False,
            mlp_activation=ActivationType.GELU,
            mlp_gated=False,
            norm_type=NormType.LAYER_NORM,
            sm_count=sm_count,
        )

    @classmethod
    def from_mistral_7b(cls, sm_count: int = 132) -> "ModelConfig":
        """Create config for Mistral 7B"""
        return cls(
            num_layers=32,
            hidden_dim=4096,
            intermediate_dim=14336,
            vocab_size=32000,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=32768,
            attention_type=AttentionType.GQA,
            has_sliding_window=True,
            sliding_window_size=4096,
            sm_count=sm_count,
        )


# ============================================================================
# Generic Instruction
# ============================================================================

@dataclass
class GenericInstruction:
    """Matches C++ GenericInstruction struct (64 bytes)"""

    opcode: int
    flags: int = 0
    layer_idx: int = 0

    # Dimensions
    m_dim: int = 0
    n_dim: int = 0
    k_dim: int = 0

    block_idx_m: int = 0
    block_idx_n: int = 0
    block_idx_k: int = 0

    # Memory offsets
    input_offset_0: int = 0
    input_offset_1: int = 0
    input_offset_2: int = 0
    output_offset: int = 0
    weight_offset: int = 0
    scratch_offset: int = 0

    # Configuration
    head_config: int = 0
    reduction_factor: int = 0
    seq_pos: int = 0
    batch_idx: int = 0

    # Sync
    dependency_mask: int = 0
    sync_slot: int = 0
    sync_count: int = 0
    parent_instr_id: int = 0

    # Metadata
    instruction_id: int = 0
    scale_factor: float = 1.0

    def serialize(self) -> np.ndarray:
        """Serialize to 32 int32 array (128 bytes, padded from 64)"""
        buffer = np.zeros(32, dtype=np.int32)

        # Pack into uint32 words
        buffer[0] = (self.opcode & 0xFF) | \
                   ((self.flags & 0xFF) << 8) | \
                   ((self.layer_idx & 0xFFFF) << 16)

        buffer[1] = (self.m_dim & 0xFFFF) | ((self.n_dim & 0xFFFF) << 16)
        buffer[2] = (self.k_dim & 0xFFFF) | ((self.block_idx_m & 0xFFFF) << 16)
        buffer[3] = (self.block_idx_n & 0xFFFF) | ((self.block_idx_k & 0xFFFF) << 16)

        buffer[4] = self.input_offset_0
        buffer[5] = self.input_offset_1
        buffer[6] = self.input_offset_2
        buffer[7] = self.output_offset
        buffer[8] = self.weight_offset
        buffer[9] = self.scratch_offset

        buffer[10] = (self.head_config & 0xFFFF) | ((self.reduction_factor & 0xFFFF) << 16)
        buffer[11] = (self.seq_pos & 0xFFFF) | ((self.batch_idx & 0xFFFF) << 16)

        buffer[12] = (self.dependency_mask & 0xFFFF) | \
                    ((self.sync_slot & 0xFF) << 16) | \
                    ((self.sync_count & 0xFF) << 24)
        buffer[13] = self.parent_instr_id

        buffer[14] = self.instruction_id
        buffer[15] = np.float32(self.scale_factor).view(np.int32)

        # Remaining slots are padding (zeros)

        return buffer

    def cost(self, model_cfg: ModelConfig) -> int:
        """Estimate computational cost (FLOPs or memory ops)"""
        if self.opcode == Opcode.MATMUL:
            return self.m_dim * self.n_dim * self.k_dim * 2
        elif self.opcode == Opcode.MATVEC:
            return self.m_dim * self.k_dim * 2
        elif self.opcode in [Opcode.RMS_NORM, Opcode.LAYER_NORM]:
            return self.m_dim * self.n_dim * 3
        elif self.opcode == Opcode.ATTENTION_PARTIAL:
            return self.m_dim * self.n_dim * self.k_dim * 4
        else:
            return self.m_dim * self.n_dim


# ============================================================================
# Universal Scheduler
# ============================================================================

class UniversalScheduler:
    """Model-agnostic instruction scheduler"""

    def __init__(self, model_cfg: ModelConfig):
        self.cfg = model_cfg
        self.instructions: List[GenericInstruction] = []
        self.next_instr_id = 0

    def build_transformer_layer(
        self,
        layer_idx: int,
        use_fused_ops: bool = True
    ) -> List[GenericInstruction]:
        """Build instruction sequence for one transformer layer"""
        instructions = []

        # 1. Pre-attention norm + QKV projection
        if use_fused_ops:
            instructions.append(self._make_fused_norm_qkv_rope(layer_idx))
        else:
            instructions.append(self._make_norm(layer_idx, norm_idx=0))
            instructions.extend(self._make_qkv_projection(layer_idx))
            if self.cfg.has_rope:
                instructions.extend(self._make_rope_embedding(layer_idx))

        # 2. Attention
        instructions.extend(self._make_attention(layer_idx))

        # 3. O projection + residual
        instructions.extend(self._make_o_projection(layer_idx))

        # 4. Pre-MLP norm + MLP
        instructions.append(self._make_norm(layer_idx, norm_idx=1))
        instructions.extend(self._make_mlp(layer_idx))

        return instructions

    def build_full_model(
        self,
        use_fused_ops: bool = True
    ) -> List[GenericInstruction]:
        """Build complete instruction sequence for the model"""
        instructions = []

        # Build all transformer layers
        for layer_idx in range(self.cfg.num_layers):
            layer_instructions = self.build_transformer_layer(layer_idx, use_fused_ops)
            instructions.extend(layer_instructions)

        # Final norm + LM head
        instructions.append(self._make_norm(self.cfg.num_layers, norm_idx=0))
        instructions.extend(self._make_lm_head())

        return instructions

    # ========== Instruction Builders ==========

    def _make_norm(self, layer_idx: int, norm_idx: int) -> GenericInstruction:
        """Create normalization instruction"""
        opcode = Opcode.RMS_NORM if self.cfg.norm_type == NormType.RMS_NORM else Opcode.LAYER_NORM

        return GenericInstruction(
            opcode=opcode,
            instruction_id=self._next_id(),
            layer_idx=layer_idx,
            m_dim=1,
            n_dim=self.cfg.hidden_dim,
            input_offset_0=0,  # hidden_states
            weight_offset=layer_idx * 2 + norm_idx,
            output_offset=0,
            scale_factor=self.cfg.norm_eps,
        )

    def _make_qkv_projection(self, layer_idx: int) -> List[GenericInstruction]:
        """Create QKV projection instructions (may be tiled)"""
        num_blocks = (self.cfg.qkv_dim + self.cfg.matmul_block_n - 1) // self.cfg.matmul_block_n

        instructions = []
        for block_idx in range(num_blocks):
            instr = GenericInstruction(
                opcode=Opcode.MATMUL,
                instruction_id=self._next_id(),
                layer_idx=layer_idx,
                m_dim=1,  # Single token (latency mode)
                n_dim=min(self.cfg.matmul_block_n, self.cfg.qkv_dim - block_idx * self.cfg.matmul_block_n),
                k_dim=self.cfg.hidden_dim,
                block_idx_n=block_idx,
                input_offset_0=0,
                weight_offset=0,  # QKV weights for this layer
                output_offset=block_idx * self.cfg.matmul_block_n,
            )
            instructions.append(instr)

        return instructions

    def _make_rope_embedding(self, layer_idx: int) -> List[GenericInstruction]:
        """Create RoPE embedding instructions"""
        # Apply RoPE to Q and K
        return [
            GenericInstruction(
                opcode=Opcode.ROPE_EMBED,
                instruction_id=self._next_id(),
                layer_idx=layer_idx,
                m_dim=self.cfg.num_q_heads,
                n_dim=1,
                k_dim=self.cfg.head_dim,
                input_offset_0=0,  # Q
                input_offset_1=1,  # K
                scale_factor=self.cfg.rope_theta,
            )
        ]

    def _make_attention(self, layer_idx: int) -> List[GenericInstruction]:
        """Create attention instructions (may be split for flash attention)"""
        instructions = []

        # For single-token decode, simple attention
        instr = GenericInstruction(
            opcode=Opcode.ATTENTION_PARTIAL,
            instruction_id=self._next_id(),
            layer_idx=layer_idx,
            m_dim=self.cfg.num_q_heads,
            n_dim=1,  # Query seq len (decode)
            k_dim=self.cfg.head_dim,
            head_config=self.cfg.attention_type,
            input_offset_0=0,  # Q
            input_offset_1=1,  # K cache
            input_offset_2=2,  # V cache
            output_offset=0,
            scale_factor=self.cfg.attn_scale,
        )
        instructions.append(instr)

        return instructions

    def _make_o_projection(self, layer_idx: int) -> List[GenericInstruction]:
        """Create O projection + residual add"""
        return [
            GenericInstruction(
                opcode=Opcode.MATMUL,
                instruction_id=self._next_id(),
                layer_idx=layer_idx,
                m_dim=1,
                n_dim=self.cfg.hidden_dim,
                k_dim=self.cfg.q_dim,
                input_offset_0=0,  # Attention output
                weight_offset=0,   # O projection weights
                output_offset=0,
            ),
            GenericInstruction(
                opcode=Opcode.RESIDUAL_ADD,
                instruction_id=self._next_id(),
                m_dim=1,
                n_dim=self.cfg.hidden_dim,
                input_offset_0=0,  # O proj output
                input_offset_1=1,  # Residual
                output_offset=0,
            )
        ]

    def _make_mlp(self, layer_idx: int) -> List[GenericInstruction]:
        """Create MLP instructions"""
        if self.cfg.mlp_gated:
            # SwiGLU or GeGLU style
            return [
                GenericInstruction(
                    opcode=Opcode.FUSED_GATE_ACT,
                    instruction_id=self._next_id(),
                    layer_idx=layer_idx,
                    m_dim=1,
                    n_dim=self.cfg.intermediate_dim,
                    k_dim=self.cfg.hidden_dim,
                    input_offset_0=0,
                    weight_offset=0,  # Gate + Up weights
                    output_offset=0,
                ),
                GenericInstruction(
                    opcode=Opcode.MATMUL,
                    instruction_id=self._next_id(),
                    layer_idx=layer_idx,
                    m_dim=1,
                    n_dim=self.cfg.hidden_dim,
                    k_dim=self.cfg.intermediate_dim,
                    input_offset_0=0,
                    weight_offset=0,  # Down weights
                    output_offset=0,
                ),
                GenericInstruction(
                    opcode=Opcode.RESIDUAL_ADD,
                    instruction_id=self._next_id(),
                    m_dim=1,
                    n_dim=self.cfg.hidden_dim,
                    input_offset_0=0,
                    input_offset_1=1,
                    output_offset=0,
                )
            ]
        else:
            # Standard MLP
            return [
                GenericInstruction(opcode=Opcode.MATMUL, instruction_id=self._next_id()),
                GenericInstruction(opcode=Opcode.GELU, instruction_id=self._next_id()),
                GenericInstruction(opcode=Opcode.MATMUL, instruction_id=self._next_id()),
                GenericInstruction(opcode=Opcode.RESIDUAL_ADD, instruction_id=self._next_id()),
            ]

    def _make_lm_head(self) -> List[GenericInstruction]:
        """Create LM head projection"""
        num_blocks = (self.cfg.vocab_size + self.cfg.matmul_block_n - 1) // self.cfg.matmul_block_n

        instructions = []
        for block_idx in range(num_blocks):
            instr = GenericInstruction(
                opcode=Opcode.MATMUL,
                instruction_id=self._next_id(),
                m_dim=1,
                n_dim=min(self.cfg.matmul_block_n, self.cfg.vocab_size - block_idx * self.cfg.matmul_block_n),
                k_dim=self.cfg.hidden_dim,
                block_idx_n=block_idx,
                input_offset_0=0,
                weight_offset=0,
                output_offset=block_idx * self.cfg.matmul_block_n,
            )
            instructions.append(instr)

        return instructions

    def _make_fused_norm_qkv_rope(self, layer_idx: int) -> GenericInstruction:
        """Create fused norm + QKV + RoPE instruction (best performance)"""
        return GenericInstruction(
            opcode=Opcode.FUSED_NORM_QKV_ROPE,
            instruction_id=self._next_id(),
            layer_idx=layer_idx,
            m_dim=1,
            n_dim=self.cfg.qkv_dim,
            k_dim=self.cfg.hidden_dim,
            input_offset_0=0,
            weight_offset=0,
            output_offset=0,
            scale_factor=self.cfg.norm_eps,
        )

    def _next_id(self) -> int:
        id_val = self.next_instr_id
        self.next_instr_id += 1
        return id_val


# ============================================================================
# Example Usage
# ============================================================================

def demo_multi_model_support():
    """Demonstrate instruction generation for different models"""

    print("=" * 80)
    print("Generic Instruction Set - Multi-Model Demo")
    print("=" * 80)

    # Test different model architectures
    models = [
        ("Llama 3.2 1B", ModelConfig.from_llama_3_1b()),
        ("GPT-2 124M", ModelConfig.from_gpt2()),
        ("Mistral 7B", ModelConfig.from_mistral_7b()),
    ]

    for model_name, config in models:
        print(f"\n{model_name}:")
        print(f"  Architecture: {config.attention_type.name}, "
              f"{config.num_q_heads}Q/{config.num_kv_heads}KV heads, "
              f"dim={config.hidden_dim}")
        print(f"  MLP: {config.mlp_activation.name}, "
              f"Norm: {config.norm_type.name}")

        scheduler = UniversalScheduler(config)

        # Build single layer instructions
        layer_instructions = scheduler.build_transformer_layer(layer_idx=0, use_fused_ops=True)

        print(f"  Instructions per layer: {len(layer_instructions)}")
        print(f"  First 3 instructions:")
        for i, instr in enumerate(layer_instructions[:3]):
            opcode_name = Opcode(instr.opcode).name
            print(f"    {i}: {opcode_name} "
                  f"(dims: {instr.m_dim}x{instr.n_dim}x{instr.k_dim})")

        # Estimate total compute
        total_cost = sum(instr.cost(config) for instr in layer_instructions)
        print(f"  Estimated FLOPs per layer: {total_cost / 1e6:.2f}M")


if __name__ == "__main__":
    demo_multi_model_support()
