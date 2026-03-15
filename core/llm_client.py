import requests
from config import LLM_BACKEND, OLLAMA_MODEL, OLLAMA_HOST, GEMINI_API_KEY, GEMINI_MODEL, DEEPSEEK_MODEL, DEEPSEEK_API_KEY
from openai import OpenAI


class LLMClient:
    def __init__(self):
        self.backend = LLM_BACKEND  # "ollama" or "gemini"

    def chat(self, system: str, user: str, temperature: float = 1.3) -> str:
        if self.backend == "gemini":
            return self._gemini(system, user, temperature)
        elif self.backend == "deepseek":
            return self._deepseek(system, user, temperature)
        else:
            return self._ollama(system, user, temperature)

    def stream_chat(self, system: str, user: str, temperature: float = 1.3):
        """Generator that yields chunks of the response."""
        if self.backend == "deepseek":
            yield from self._deepseek_stream(system, user, temperature)
        elif self.backend == "ollama":
            yield from self._ollama_stream(system, user, temperature)
        else:
            # Fallback: yield complete response at once
            result = self.chat(system, user, temperature)
            yield result

    # ── Ollama ────────────────────────────────────────────────────
    def _ollama(self, system: str, user: str, temperature: float) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _ollama_stream(self, system: str, user: str, temperature: float):
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": True,
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120, stream=True)
        r.raise_for_status()
        for chunk in r.iter_lines():
            if chunk:
                data = chunk.decode('utf-8')
                if data.startswith('data:'):
                    data = data[5:].strip()
                if data and data != '[DONE]':
                    try:
                        import json
                        content = json.loads(data).get("message", {}).get("content", "")
                        if content:
                            yield content
                    except:
                        pass

    # ── Gemini ────────────────────────────────────────────────────
    def _gemini(self, system: str, user: str, temperature: float) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models"
            f"/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "system_instruction": {
                "parts": [{"text": system}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": user}]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 2048,
            },
        }
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _deepseek(self, system: str, user: str, temperature=1.3) -> str:

        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            stream=False,
            temperature=temperature,
            max_tokens=2048,
        )

        return response.choices[0].message.content

    def _deepseek_stream(self, system: str, user: str, temperature=1.3):
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            stream=True,
            temperature=temperature,
            max_tokens=2048,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

