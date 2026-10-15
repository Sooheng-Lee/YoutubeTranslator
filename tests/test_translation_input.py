from types import SimpleNamespace

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
