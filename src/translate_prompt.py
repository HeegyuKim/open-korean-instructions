import fire
from .language_models.utils import load_model
from .language_models.mt_model import Seagull13B
from .utils.eval_utils import batched_iteration, estimate_skip_length
from datasets import load_dataset, Dataset
from tqdm import tqdm
import jsonlines
import time
from pprint import pprint



def load_target_dataset(name):
    if name == "openbmb/UltraInteract_pair":
        ds = load_dataset("openbmb/UltraInteract_pair", split="train")
        df = ds.to_pandas()
        df["instruction"] = df.trajectory.apply(lambda x: x[0]['value'])
        df = df.drop_duplicates(subset=["instruction"])
        df = Dataset.from_pandas(df)
        print(len(df), "items in dataset")
        return df, ["instruction"]
    elif name == "openbmb/UltraFeedback":
        ds = load_dataset("openbmb/UltraFeedback", split="train")
        df = ds.to_pandas()
        df = df.drop_duplicates(subset=["instruction"])
        df = Dataset.from_pandas(df)
        print(len(df), "items in dataset")
        return df, ["instruction"]
    else:
        raise ValueError(f"Unknown dataset: {name}")

def main(
    model: str,
    output_file: str = None,
    chat_template: str = None,
    dataset: str = "openbmb/UltraInteract_pair",
    prompt_length: int = 2048,
    max_length: int = 4096,
    batch_size: int = 2,
    eos_token: str = None,
):
    model_name = model
    dataset_name = dataset
    if output_file is None:
        basename = f"{model}---{dataset}".replace('/', '_')
        output_file = f"./data/{basename}"

    gen_args = dict(do_sample=True, temperature=0.7, top_p=0.95, top_k=50, max_new_tokens=max_length-prompt_length)
    dataset, keys = load_target_dataset(dataset)
    model = load_model(model, prompt_length=prompt_length, max_length=max_length, batch_size=batch_size, gen_args=gen_args, chat_template=chat_template, eos_token=eos_token)
    

    skip_length = estimate_skip_length(output_file)
    if skip_length > 0:
        dataset = dataset.select(range(skip_length, len(dataset)))
        print(f"Skipping {skip_length} examples")

    fout = jsonlines.open(output_file, "a")

    for batch in tqdm(batched_iteration(dataset, batch_size=batch_size), total=len(dataset)//batch_size):
        for key in keys:
            prompts = ["Hello! " + x[key] for x in batch]
            histories = [[{"role": "system", "content": "주어진 문장을 한국어로 번역하세요."}] for _ in range(len(batch))]
            responses = model.generate_batch(prompts, histories=histories, gen_args=gen_args, generation_prefix="안녕하세요!")

            for item, response in zip(batch, responses):
                item[f"{key}_ko"] = response
    
        for item, response in zip(batch, responses):
            item["generator"] = model_name
            item["dataset_name"] = dataset_name
            fout.write(item)

if __name__ == "__main__":
    fire.Fire(main)