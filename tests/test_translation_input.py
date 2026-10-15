from types import SimpleNamespace
import pytest

from app.services.translategemma_translator import TranslateGemmaTranslator


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        assert len(content) == 1
        assert content[0]["source_lang_code"] == "en"
        assert content[0]["target_lang_code"] == "ko"
        # Character-count stand-in to exercise recursion and template overhead.
        return {"input_ids": SimpleNamespace(shape=(1, len(content[0]["text"]) + 100))}


def test_long_input_is_split_without_truncating_source():
    translator = TranslateGemmaTranslator()
    translator.processor = Processor()
    text = ("A long sentence. " * 400) + "마지막"
    chunks = translator.split(text, "en", "ko")
    assert "".join(chunks) == text
    assert len(chunks) > 1
    assert all(len(chunk) + 100 <= 1900 for chunk in chunks)


@pytest.mark.parametrize("configured,tokenizer_pad,eos,expected", [
    (7, 0, 1, 7), (None, 0, 1, 0), (None, None, 1, 1),
    (None, None, [1, 107], 1),
])
def test_generation_pad_resolution_preserves_zero_and_eos(configured, tokenizer_pad, eos, expected):
    translator = TranslateGemmaTranslator()
    translator.model = SimpleNamespace(generation_config=SimpleNamespace(pad_token_id=configured, eos_token_id=eos))
    translator.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=tokenizer_pad))
    assert translator._pad_token_id() == expected
    assert translator.model.generation_config.eos_token_id == eos


def test_pad_can_come_from_text_model_config():
    translator = TranslateGemmaTranslator()
    translator.model = SimpleNamespace(generation_config=SimpleNamespace(pad_token_id=None, eos_token_id=1),
                                      config=SimpleNamespace(text_config=SimpleNamespace(pad_token_id=0)))
    translator.processor = SimpleNamespace()
    assert translator._pad_token_id() == 0


@pytest.mark.parametrize("precision", ["4bit", "bfloat16", "float16"])
def test_gemma_cuda_uses_bf16_instead_of_overflow_prone_fp16(precision):
    fake_torch = SimpleNamespace(float32="fp32", bfloat16="bf16",
                                 cuda=SimpleNamespace(is_bf16_supported=lambda: True))
    assert TranslateGemmaTranslator._cuda_dtype(precision, fake_torch) == "bf16"


def test_unsupported_bf16_fails_explicitly():
    from app.services.translategemma_translator import TranslationEngineError
    fake_torch = SimpleNamespace(float32="fp32", cuda=SimpleNamespace(is_bf16_supported=lambda: False))
    with pytest.raises(TranslationEngineError, match="BF16"):
        TranslateGemmaTranslator._cuda_dtype("4bit", fake_torch)
    assert TranslateGemmaTranslator._cuda_dtype("float32", fake_torch) == "fp32"


def test_invalid_logits_abort_before_generating_pad_forever():
    import av  # initialize media DLLs first on Windows
    import torch
    from app.services.translategemma_translator import TranslationEngineError
    for values in ([float("nan"), 0], [float("inf"), 0], [-float("inf"), -float("inf")]):
        with pytest.raises(TranslationEngineError, match="NaN/Inf"):
            TranslateGemmaTranslator._validate_logits(torch.tensor([values]))
    masked = torch.tensor([[0.2, -float("inf"), 0.1]])
    assert TranslateGemmaTranslator._validate_logits(masked) is masked
