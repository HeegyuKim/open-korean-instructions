
from .base import BaseLanguageModel
from .judge_model import BaseJudge

class OpenAIModel(BaseLanguageModel, BaseJudge):

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        super().__init__()
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
    
    def generate(self, prompt, history = None, generation_prefix: str = None, gen_args = {}):
        if history is None:
            history = []
        history.append({
            'role': 'user',
            'content': prompt,
        })
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        output = response.choices[0].message.content
        return output

    def moderate(self, instruction, response = None, gen_args: dict = None):
        if response:
            result = self.client.moderations.create(input=f"Q: {instruction}\nA: {response}")
        else:
            result = self.client.moderations.create(input=instruction)

        return result.model_dump()
        # if return_dict:
        #     return result.model_dump()
        # else:
        #     return result.model_dump()['result'][0]['flagged']