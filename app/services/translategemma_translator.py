import gc
from pathlib import Path
import re

from app.models import MODEL_ID


class TranslationEngineError(RuntimeError):
    """An engine-wide error for which retrying every subtitle is not useful."""


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
            compute_dtype = self._cuda_dtype(settings.precision, torch)
            kwargs.update(device_map={"": 0}, dtype=compute_dtype)
            if settings.precision == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype)
            report(f"TranslateGemma 실행: {settings.precision} / 계산 {compute_dtype}", -1)
        if model_path == MODEL_ID:
            from app.services.llm_model_downloader import LLMModelDownloader
            model_path = str(LLMModelDownloader().download(model_path, self.access_token, token, report))
        token.check()
        self.processor = AutoProcessor.from_pretrained(model_path)
        token.check()
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        self.model.eval()
        token.check()

    @staticmethod
    def _cuda_dtype(precision, torch):
        if precision == "float32":
            return torch.float32
        # Gemma 3's BF16 weights/activations can overflow in FP16, including NF4
        # models whose remaining layers and quantized matmuls still use FP16.
        if not torch.cuda.is_bf16_supported():
            raise TranslationEngineError("TranslateGemma의 안정적인 CUDA 실행에는 BF16 지원 GPU가 필요합니다. 실행 장치와 모델 환경을 확인하세요.")
        return torch.bfloat16

    @staticmethod
    def _validate_logits(scores):
        import torch
        if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any(dim=-1).all():
            raise TranslationEngineError("번역 모델 계산에서 NaN/Inf가 발생했습니다. 앱을 재시작하여 4bit/BF16 설정으로 재시도하세요.")
        return scores

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

    def _pad_token_id(self):
        generation = self.model.generation_config
        if generation.pad_token_id is not None:
            return generation.pad_token_id
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is not None:
            return pad
        config = getattr(self.model, "config", None)
        for candidate in (config, getattr(config, "text_config", None)):
            pad = getattr(candidate, "pad_token_id", None)
            if pad is not None:
                return pad
        eos = generation.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else None
        if eos is None:
            raise ValueError("모델의 PAD/EOS 토큰 설정이 없습니다.")
        return eos

    def translate(self, text, source, target, token):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        class ValidateLogits(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return TranslateGemmaTranslator._validate_logits(scores)

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
                                                pad_token_id=self._pad_token_id(),
                                                top_k=None, top_p=None,
                                                logits_processor=LogitsProcessorList([ValidateLogits()]),
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
