import os
from dotenv import load_dotenv
from openai import OpenAI

class APIClient():
    def __init__(self, url: str, api_key: str) -> str:
        assert url, "url must be non-empty and non-null"
        assert api_key, "api key must be non-empty and non-null"

        self.client = OpenAI(
            base_url=url,
            api_key=api_key
        )

load_dotenv()

url = os.getenv("URL")
api_key = os.getenv("API_KEY")

api_client = APIClient(url=url, api_key=api_key)