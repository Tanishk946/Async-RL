from __future__ import annotations

INSTRUCTION = (
    "Compute {a} + {b}.\n"
    "Put the final answer on its own line as: #### <number>"
)


def addition_prompt(a: int, b: int) -> str:
    return INSTRUCTION.format(a=a, b=b)


def render_chat(user_text: str, tokenizer=None) -> str:
    """Apply the model chat template when a tokenizer is provided."""
    messages = [
        {"role": "system", "content": "You are a careful math assistant. Follow the answer format exactly."},
        {"role": "user", "content": user_text},
    ]
    if tokenizer is None:
        return (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def sample_addends(n: int, lo: int, hi: int, seed: int = 0) -> list[tuple[int, int]]:
    import random

    rng = random.Random(seed)
    return [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n)]
