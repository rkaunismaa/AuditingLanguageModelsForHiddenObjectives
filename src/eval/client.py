from openai import OpenAI

class OpenAIClient:
    def __init__(self, base_url="http://localhost:8000/v1", model="organism", api_key="x"):
        self._c = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
    def complete(self, prompt: str, max_tokens=512, temperature=0.7) -> str:
        r = self._c.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return r.choices[0].message.content
