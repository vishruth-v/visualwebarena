from text_generation import Client  # type: ignore
import os


def generate_from_huggingface_completion(
    prompt: str,
    model_endpoint: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: list[str] | None = None,
) -> str:
    client = Client(model_endpoint, headers = {"Authorization": f"Bearer {os.environ.get('HF_API_KEY')}"}, timeout=60)
    generation: str = client.generate(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop_sequences=stop_sequences,
        truncate=4096 # 8k for llama3
    ).generated_text

    return generation
