#!/usr/bin/env python3
"""Test script for EGDC tokenizer, data generator, and dataset."""

import sys
sys.path.insert(0, "/Users/bobbyprice/projects/nCPU")

from egdc.core.tokenizer import NCPUTokenizer, VOCAB_SIZE, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN
from egdc.core.data_generator import NCPUDataGenerator
from egdc.core.dataset import NCPUDataset


def test_tokenizer():
    print("=" * 60)
    print("TOKENIZER TESTS")
    print("=" * 60)

    tok = NCPUTokenizer()
    print(f"Vocab size: {tok.vocab_size} (expected 346)")
    assert tok.vocab_size == 346, f"Expected 346, got {tok.vocab_size}"

    # Test special tokens
    assert MASK_TOKEN == 342
    assert PAD_TOKEN == 343
    assert BOS_TOKEN == 344
    assert EOS_TOKEN == 345
    print("Special tokens: OK")

    # Test basic encode/decode round-trip
    prog = """\
MOV_IMM R0 42
MOV_IMM R1 10
ADD R2 R1
HALT"""
    tokens = tok.encode(prog)
    print(f"\nOriginal program:\n{prog}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)} (BOS + {(len(tokens)-2)//4} instructions * 4 + EOS)")

    decoded = tok.decode(tokens)
    print(f"\nDecoded:\n{decoded}")

    # Round-trip test
    re_encoded = tok.encode(decoded)
    assert tokens == re_encoded, f"Round-trip failed!\n{tokens}\n!=\n{re_encoded}"
    print("\nRound-trip: PASS")

    # Test all opcodes
    all_opcodes_prog = """\
NOP
MOV_IMM R0 255
MOV_REG R1 R0
ADD R2 R3
SUB R4 R5
MUL R6 R7
AND R0 R1
OR R2 R3
XOR R4 R5
CMP R6 R7
BEQ 0
BNE 10
BGT 63
HALT"""
    tokens = tok.encode(all_opcodes_prog)
    decoded = tok.decode(tokens)
    re_encoded = tok.encode(decoded)
    assert tokens == re_encoded, "All-opcodes round-trip failed!"
    print("All opcodes round-trip: PASS")

    # Test padding
    padded = tok.pad(tokens, 128)
    assert len(padded) == 128
    assert padded[-1] == PAD_TOKEN
    print(f"Padding to 128: PASS (last token = PAD={padded[-1]})")

    # Test instruction count
    ic = tok.instruction_count(tokens)
    assert ic == 14, f"Expected 14 instructions, got {ic}"
    print(f"Instruction count: {ic} PASS")

    print("\nAll tokenizer tests PASSED!")


def test_generator():
    print("\n" + "=" * 60)
    print("DATA GENERATOR TESTS")
    print("=" * 60)

    gen = NCPUDataGenerator(seed=123)
    print(f"Number of templates: {gen.num_templates}")
    print(f"Template names: {gen.template_names}")
    assert gen.num_templates >= 20, f"Expected >= 20 templates, got {gen.num_templates}"

    tok = NCPUTokenizer()

    # Test each template
    for name in gen.template_names:
        spec, tokens = gen.generate_one(template_name=name)
        decoded = tok.decode(tokens)
        # Verify round-trip
        re_encoded = tok.encode(decoded)
        assert tokens == re_encoded, f"Round-trip failed for template '{name}'!\n{decoded}"
        n_instr = tok.instruction_count(tokens)
        n_cases = len(spec["test_cases"])
        print(f"  {name:20s}: {n_instr:2d} instructions, {n_cases} test cases - PASS")

    # Test batch generation
    batch = gen.generate_batch(100)
    assert len(batch) == 100
    print(f"\nBatch of 100: OK")

    # Test uniqueness with a larger sample
    samples = gen.generate_batch(1000)
    unique_progs = set(tuple(t) for _, t in samples)
    print(f"1000 samples -> {len(unique_progs)} unique programs ({len(unique_progs)/10:.1f}% unique)")
    assert len(unique_progs) > 500, "Expected >50% unique programs"

    print("\nAll generator tests PASSED!")


def test_dataset():
    print("\n" + "=" * 60)
    print("DATASET TESTS")
    print("=" * 60)

    ds = NCPUDataset(num_samples=1000, seq_len=128, spec_len=32, seed=42)
    print(f"Dataset size: {len(ds)}")
    print(f"Vocab size: {ds.vocab_size}")
    assert len(ds) == 1000

    # Test a few items
    for i in [0, 1, 500, 999]:
        masked, mask_pos, original, spec, timestep = ds[i]
        assert masked.shape == (128,), f"masked shape: {masked.shape}"
        assert mask_pos.shape == (128,), f"mask_pos shape: {mask_pos.shape}"
        assert original.shape == (128,), f"original shape: {original.shape}"
        assert spec.shape == (32,), f"spec shape: {spec.shape}"
        assert 0.0 <= timestep.item() <= 1.0, f"timestep: {timestep}"

        # Verify masking is consistent
        n_masked = mask_pos.sum().item()
        for j in range(128):
            if mask_pos[j]:
                assert masked[j].item() == MASK_TOKEN
            else:
                assert masked[j].item() == original[j].item()

        print(f"  Item {i:4d}: timestep={timestep:.3f}, masked={n_masked:3d}/128 tokens - OK")

    # Test decode
    _, _, original, _, _ = ds[0]
    asm = ds.decode_program(original)
    print(f"\nSample decoded program:\n{asm}")

    # Test DataLoader
    dl = ds.get_dataloader(batch_size=16)
    batch = next(iter(dl))
    masked_b, mask_b, orig_b, spec_b, t_b = batch
    assert masked_b.shape == (16, 128)
    assert spec_b.shape == (16, 32)
    assert t_b.shape == (16,)
    print(f"\nDataLoader batch shapes: masked={masked_b.shape}, spec={spec_b.shape}, t={t_b.shape}")

    print("\nAll dataset tests PASSED!")


if __name__ == "__main__":
    test_tokenizer()
    test_generator()
    test_dataset()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
