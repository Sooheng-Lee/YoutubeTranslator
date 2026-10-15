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
