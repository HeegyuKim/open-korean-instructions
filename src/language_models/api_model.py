
from .base import BaseLanguageModel
from .judge_model import BaseJudge
import os
import retry



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

class GeminiModel(BaseLanguageModel):

    def __init__(self, model: str = "gemini-1.5-flash") -> None:
        super().__init__()
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model)

    @retry.retry(tries=3, delay=10)
    def generate(self, prompt, history = None, generation_prefix: str = None, gen_args = {}):
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        if generation_prefix:
            raise NotImplementedError("Generation prefix is not supported by Gemini model")
        
        generation_config=genai.GenerationConfig(
            max_output_tokens=gen_args.get("max_new_tokens", 128),
            temperature=gen_args.get("temperature", 1.0),
        )
        response = self.model.generate_content(
            prompt, 
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                # HarmCategory.HARM_CATEGORY_TOXICITY: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        print(response)
        try:
            return response.text
        except:
            return f"Gemini Refusal!!! {response.prompt_feedback}"

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