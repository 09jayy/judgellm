from openai import OpenAI

class APIClient():
    def __init__(self, url: str, api_key: str) -> None:
        assert url, "url must be non-empty and non-null"
        assert api_key, "api key must be non-empty and non-null"

        self._client = OpenAI(
            base_url=url,
            api_key=api_key
        )

    def call_api(
            self,
            model_id: str,
            conversation: list[dict[str,str]], 
            max_tokens: int = 36
        ) -> str: 
        
        completion = self._client.chat.completions.create(
            extra_body={},
            model=model_id,
            messages=conversation,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content

"""
# Example usage
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("URL")
api_key = os.getenv("API_KEY")

api_client = APIClient(url=url, api_key=api_key)

GPT = GPT3.5(
        executor=APIExecutor(
            model_id="openai/gpt-3.5-turbo",
            api_client=api_client # same api_client for multiple classifiers are possible
        )
    )
"""