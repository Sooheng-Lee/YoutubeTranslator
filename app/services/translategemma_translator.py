import gc
from pathlib import Path
import re

from app.models import MODEL_ID


class TranslateGemmaTranslator:
    def __init__(self, access_token=None):
        self.model = None
        self.processor = None
        self.access_token = access_token or None

    def load(self, settings, token, report):
        import av  # noqa: F401 -- initialize FFmpeg DLLs before torch on Windows
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        token.check()
        model_path = settings.model_path.strip()
        if model_path != MODEL_ID:
            model_path = str(Path(model_path).expanduser())
            from transformers import AutoConfig
            if not Path(model_path).is_dir():
                raise ValueError("TranslateGemma 12B ID 또는 해당 모델의 로컬 폴더를 지정하세요.")
            config = AutoConfig.from_pretrained(model_path)
            # Gemma 3 12B text backbone: 48 layers / hidden size 3840.
            text = getattr(config, "text_config", config)
            if getattr(text, "hidden_size", None) != 3840 or getattr(text, "num_hidden_layers", None) != 48:
                raise ValueError("로컬 모델이 Gemma 3 12B 구조와 일치하지 않습니다.")
        report("TranslateGemma 12B 모델 준비 (최초 다운로드는 오래 걸릴 수 있음)", -1)
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. CPU를 선택하거나 CUDA PyTorch 설치를 확인하세요.")
        if settings.device == "cpu" and settings.precision != "float32":
            raise ValueError("CPU 실행은 float32를 선택하세요. 12B 모델은 많은 시스템 RAM을 사용합니다.")
        kwargs = {"low_cpu_mem_usage": True, "dtype": torch.float32}
        if settings.device == "cuda":
            kwargs.update(device_map={"": 0}, dtype=torch.float32 if settings.precision == "float32" else torch.float16)
            if settings.precision == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16)
        if model_path == MODEL_ID:
            from app.services.llm_model_downloader import LLMModelDownloader
            model_path = str(LLMModelDownloader().download(model_path, self.access_token, token, report))
        token.check()
        self.processor = AutoProcessor.from_pretrained(model_path)
        token.check()
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        self.model.eval()
        token.check()

    def _messages(self, text, source, target):
        return [{"role": "user", "content": [{"type": "text", "source_lang_code": source,
                   "target_lang_code": target, "text": text}]}]

    def _input(self, text, source, target):
        return self.processor.apply_chat_template(self._messages(text, source, target), tokenize=True,
                                                  add_generation_prompt=True, return_dict=True, return_tensors="pt")

    def split(self, text, source, target):
        # Include the actual template overhead in the 2K input budget.
        if self._input(text, source, target)["input_ids"].shape[-1] <= 1900:
            return [text]
        if len(text) < 2:
            raise ValueError("모델 입력 한도를 초과했습니다.")
        boundaries = [m.end() for m in re.finditer(r"[.!?。！？]\s*|\s+", text)]
        midpoint = len(text) // 2
        usable = [n for n in boundaries if 0 < n < len(text)]
        cut = min(usable, key=lambda n: abs(n - midpoint)) if usable else midpoint
        return self.split(text[:cut], source, target) + self.split(text[cut:], source, target)

    def translate(self, text, source, target, token):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnCancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return token.cancelled

        outputs = []
        for chunk in self.split(text, source, target):
            token.check()
            inputs = self._input(chunk, source, target).to(self.model.device)
            length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                                stopping_criteria=StoppingCriteriaList([StopOnCancel()]))
            token.check()
            ids = generated[0][length:]
            eos = self.model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else (eos or [])
            if len(ids) >= 1024 and int(ids[-1]) not in eos:
                raise RuntimeError("출력 길이 한도로 번역이 잘렸습니다. 해당 구간을 재시도하세요.")
            result = self.processor.decode(ids, skip_special_tokens=True).strip()
            if not result:
                raise RuntimeError("빈 번역 결과가 반환되었습니다.")
            outputs.append(result)
        return " ".join(outputs)

    def unload(self):
        self.model = None
        self.processor = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
