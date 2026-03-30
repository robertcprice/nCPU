import torch

from egdc.mog_tokenizer import MogCodeTokenizer, MASK_TOKEN


def test_mask_function_bodies_preserves_signatures_and_braces():
    from egdc.mog_completion import mask_function_bodies

    code = '''
fn add(a: i64, b: i64) -> i64 {
    return a + b;
}

fn main() -> i64 {
    println_i64(add(2, 3));
    return 0;
}
'''
    scaffold = mask_function_bodies(code)

    assert "fn add(a: i64, b: i64) -> i64 {" in scaffold
    assert "fn main() -> i64 {" in scaffold
    assert "return a + b;" not in scaffold
    assert "println_i64(add(2, 3));" not in scaffold
    assert scaffold.count("{") == code.count("{")
    assert scaffold.count("}") == code.count("}")


def test_build_completion_tokens_keeps_prefix_and_masks_body():
    from egdc.mog_completion import build_completion_tokens

    tok = MogCodeTokenizer()
    code = '''fn add(a: i64, b: i64) -> i64 {
    return a + b;
}
'''
    initial_tokens, fixed_positions, original_tokens = build_completion_tokens(code, tok, seq_len=128)

    assert initial_tokens.shape == original_tokens.shape
    assert fixed_positions.shape[0] == 128
    assert fixed_positions.any().item() is True
    assert (initial_tokens == MASK_TOKEN).any().item() is True

    # signature bytes should remain fixed and equal to original
    prefix = "fn add(a: i64, b: i64) -> i64 {"
    prefix_tokens = tok.encode(prefix, add_bos_eos=False)
    # skip BOS at original_tokens[0]
    for i, t in enumerate(prefix_tokens, start=1):
        assert original_tokens[i].item() == t
        assert initial_tokens[i].item() == t
        assert fixed_positions[i].item() is True


def test_completion_generation_preserves_fixed_tokens():
    from egdc.mog_completion import complete_mog_from_initial
    from egdc.mog_model import MogMaskedDiffusion, MogDiffusionConfig

    model = MogMaskedDiffusion(MogDiffusionConfig.tiny())
    model.eval()

    initial = torch.full((1, 32), MASK_TOKEN, dtype=torch.long)
    fixed = torch.zeros((1, 32), dtype=torch.bool)
    initial[0, 0] = ord('f')
    initial[0, 1] = ord('n')
    fixed[0, 0] = True
    fixed[0, 1] = True

    out = complete_mog_from_initial(
        model=model,
        initial_tokens=initial,
        fixed_positions=fixed,
        spec_tokens=None,
        num_steps=4,
        temperature=1.0,
        device=torch.device('cpu'),
    )
    assert out.shape == initial.shape
    assert out[0, 0].item() == ord('f')
    assert out[0, 1].item() == ord('n')
