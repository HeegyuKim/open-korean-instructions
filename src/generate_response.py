import fire
from .language_models.utils import load_model
from .utils.eval_utils import batched_iteration, estimate_skip_length
from datasets import load_dataset
from tqdm import tqdm
import jsonlines
import time


def get_prompt_dataset(name):
    if name == "SJ-Donald/orca-dpo-pairs-ko":
        return load_dataset("SJ-Donald/orca-dpo-pairs-ko", split="train"), "question"
    else:
        raise ValueError(f"Unknown dataset: {name}")

def main(
    model: str,
    output_file: str = None,
    chat_template: str = None,
    dataset: str = "SJ-Donald/orca-dpo-pairs-ko",
    prompt_length: int = 1024,
    max_length: int = 2048,
    batch_size: int = 4,
    eos_token: str = None,
    api_delay: int = 3
):
    model_name = model
    dataset_name = dataset
    if output_file is None:
        basename = f"{model}---{dataset}".replace('/', '_')
        output_file = f"./data/{basename}"

    gen_args = dict(do_sample=True, temperature=0.7, top_p=0.95, top_k=50, max_new_tokens=max_length-prompt_length)
    dataset, prompt_key = get_prompt_dataset(dataset)
    model = load_model(model, prompt_length=prompt_length, max_length=max_length, batch_size=batch_size, gen_args=gen_args, chat_template=chat_template, eos_token=eos_token)
    
    skip_length = estimate_skip_length(output_file)
    if skip_length > 0:
        dataset = dataset.select(range(skip_length, len(dataset)))
        print(f"Skipping {skip_length} examples")

    fout = jsonlines.open(output_file, "a")

    for batch in tqdm(batched_iteration(dataset, batch_size=batch_size), total=len(dataset)//batch_size):
        prompts = [x[prompt_key] for x in batch]

        if "system" in batch[0]:
            if "gemma" in model_name:
                # prompts = [f"{x['system']}\n\n{x[prompt_key]}" for x in batch]
                history = None
            else:
                history = [[{"role": "system", "content": x.get("system")}] for x in batch]
        else:
            history = None

        
        responses = model.generate_batch(prompts, histories=history, gen_args=gen_args)

        for item, response in zip(batch, responses):
            item["response"] = response
            item["generator"] = model_name
            item["dataset"] = dataset_name
            fout.write(item)

        if api_delay > 0 and (model_name.startswith("openai") or model_name.startswith("google")):
            time.sleep(api_delay)

if __name__ == "__main__":
    fire.Fire(main)