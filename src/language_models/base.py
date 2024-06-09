
class BaseLanguageModel:
    def generate(self, prompt, history = None, generation_prefix: str = None, gen_params = {}):
        raise NotImplementedError()
    
    def generate_batch(self, prompts, histories = None, generation_prefix: str = None, gen_args = {}):
        if histories is None:
            return [self.generate(prompt, generation_prefix, gen_args) for prompt in prompts]
        else:
            return [self.generate(prompt, history, generation_prefix, gen_args) for prompt, history in zip(prompts, histories)]
            