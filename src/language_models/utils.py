


def load_model(
        model_name: str, 
        prompt_length: int = 1024, 
        max_length: int = 2048,
        eos_token: str = None,
        chat_template: str = None,
        batch_size: int = 1,
        gen_args: dict = None,
        ):
    if model_name.startswith("openai"):
        from .api_model import OpenAIModel
        return OpenAIModel()
    elif model_name.startswith("google"):
        from .api_model import GeminiModel
        return GeminiModel(model_name.split("/", 1)[1])
    else:
        import torch
        if torch.cuda.is_available():
            from .transformers_model import HuggingfaceModel
            return HuggingfaceModel(model_name)
        else:
            from . import flax
            from .flax.hf_models import FlaxHuggingfaceModel
            return FlaxHuggingfaceModel(
                model_name,
                prompt_length=prompt_length,
                max_length=max_length,
                fully_sharded_data_parallel=False,
                eos_token=eos_token,
                batch_size=batch_size,
                gen_args=gen_args,
                chat_template=chat_template,
                )


def load_judge_model(
        model_name: str, 
        prompt_length: int = 1024, 
        max_length: int = 2048,
        eos_token: str = None,
        batch_size: int = 1,
        gen_args: dict = None,
        ):
    import torch
    if model_name.startswith("openai"):
        from .api_model import OpenAIModel
        return OpenAIModel()
    elif model_name.startswith("clf:"):
        from .judge_model import ClassificationJudge
        return ClassificationJudge(model_name[len("clf:"):])
    else:
        if torch.cuda.is_available():
            from .transformers_model import HuggingfaceModel
            model = HuggingfaceModel(model_name)
        else:
            from .flax.hf_models import FlaxHuggingfaceModel
            model = FlaxHuggingfaceModel(
                model_name,
                prompt_length=prompt_length,
                max_length=max_length,
                fully_sharded_data_parallel=False,
                eos_token=eos_token,
                batch_size=batch_size,
                gen_args=gen_args,
                )

        from .judge_model import MDJudge
        return MDJudge(model)