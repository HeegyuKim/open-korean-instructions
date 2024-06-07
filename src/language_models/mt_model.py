

SEAGULL_PROMPT = """<|im_start|>system
주어진 문장을 한국어로 번역하세요.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

class Seagull13B:
    def __init__(self, model) -> None:
        self.model = model

    def translate_batch(self, texts: list[str]):
        prompt = [SEAGULL_PROMPT.format(instruction=x) for x in texts]
        return self.model.generat