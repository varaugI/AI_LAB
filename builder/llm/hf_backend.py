"""Optional Hugging Face backend for production-scale pretrained/fine-tuned models."""

from __future__ import annotations

from pathlib import Path
import threading

from builder.books.backends import BackendResponse, BackendUnavailable, ChatBackend


class HuggingFaceBackend(ChatBackend):
    name = "huggingface"

    def __init__(
        self,
        model_name_or_path: str,
        *,
        adapter_path: str = "",
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendUnavailable(
                "Install requirements-production.txt to use Hugging Face models."
            ) from exc
        self.torch = torch
        self.model_id = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"trust_remote_code": trust_remote_code}
        if torch.cuda.is_available():
            kwargs.update({"device_map": "auto", "torch_dtype": "auto"})
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise BackendUnavailable("Install peft to load a LoRA adapter.") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.lock = threading.Lock()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def available(self) -> bool:
        return True

    def _prompt(self, messages: list[dict]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        pieces = []
        for message in messages:
            pieces.append(f"{message.get('role', 'user').upper()}: {message.get('content', '')}")
        pieces.append("ASSISTANT:")
        return "\n\n".join(pieces)

    def generate(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 900) -> BackendResponse:
        prompt = self._prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        sampling = temperature > 0
        generation_options = {
            "max_new_tokens": int(max_tokens),
            "do_sample": sampling,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sampling:
            generation_options.update({
                "temperature": max(float(temperature), 1e-5),
                "top_p": 0.95,
            })
        with self.lock, self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_options)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            raise BackendUnavailable("The Hugging Face model returned an empty response.")
        return BackendResponse(text=text, model=self.model_id, backend=self.name)
