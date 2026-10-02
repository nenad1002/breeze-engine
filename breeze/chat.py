"""Pure Qwen3.5 chat-template helpers."""


def chat_prompt(prompt, no_thinking=False):
    """Single-user suffix from the original Qwen3.5 chat template."""
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return text + ("<think>\n\n</think>\n\n" if no_thinking else "<think>\n")