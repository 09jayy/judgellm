from typing import Protocol
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .utils import APIClient

# quantization config to optimize large 7B models for 16GB VRAM
# reduces model size with minimal different in output
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

class LLMExecutor(Protocol):
    def execute(self, conversation, max_tokens: int = 30) -> str:
        ...


class LocalExecutor(LLMExecutor):
    def __init__(self, model_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, quantization_config=quant_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def execute(self, conversation, max_tokens: int = 36) -> str:
        tokenized_input = self.tokenizer([conversation], return_tensors='pt', add_special_tokens=False).to(self.device)
        result = self.model.generate(**tokenized_input, max_new_tokens=max_tokens)
        decoded = self.tokenizer.decode(result[0][len(tokenized_input['input_ids'][0]):], skip_special_tokens=True)
        
        return decoded

class APIExecutor(LLMExecutor):
    def __init__(self, model_id: str, api_client: APIClient):
        self._model = model_id
        self._api_client = api_client # allows user to share API clients across code base when injected

    def execute(self, conversation, max_tokens: int = 36) -> str:
        return self._api_client.call_api(self._model, conversation,max_tokens)
