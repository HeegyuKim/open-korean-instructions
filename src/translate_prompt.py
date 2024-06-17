import fire
from .language_models.utils import load_model
from .utils.eval_utils import batched_iteration, estimate_skip_length
from datasets import load_dataset, Dataset
from tqdm import tqdm
import jsonlines
import time
from pprint import pprint



def load_target_dataset(name):
    if name == "openbmb/UltraInteract_sft":
        ds = load_dataset("openbmb/UltraInteract_sft", split="train")
        df = ds.to_pandas()
        df = df.drop_duplicates(subset=["instruction"])
        df = Dataset.from_pandas(df)
        print(len(df), "items in dataset")
        return df, ["instruction", "response"]
    elif name == "HuggingFaceH4/ultrachat_200k":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        flatten_items = []
        for item in ds:
            for i, turn in enumerate(item["messages"]):
                flatten_items.append(dict(
                    content=turn["content"],
                    role=turn["role"],
                    idx=i,
                    prompt_id=item["prompt_id"],
                ))
        df = Dataset.from_list(flatten_items)
        print(len(df), "items in dataset")
        return df, ["content"]
    elif name == "openbmb/UltraInteract_pair":
        ds = load_dataset("openbmb/UltraInteract_pair", split="train")
        df = ds.to_pandas()
        df["instruction"] = df.trajectory.apply(lambda x: x[0]['value'])
        df = df.drop_duplicates(subset=["instruction"])
        df = Dataset.from_pandas(df)
        print(len(df), "items in dataset")
        return df, ["instruction"]
    elif name == "HuggingFaceH4/ultrafeedback_binarized":
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        def mapper(item):
            return {
                "instruction": item["chosen"][0]['content'],
                "chosen": item["chosen"][-1]['content'],
                "rejected": item["rejected"][-1]['content'],
            }
        ds = ds.map(mapper)
        return ds, ["instruction", "rejected"]
        
    elif name == "openbmb/UltraFeedback":
        ds = load_dataset("openbmb/UltraFeedback", split="train")
        df = ds.to_pandas()
        df = df.drop_duplicates(subset=["instruction"])
        df = Dataset.from_pandas(df)
        print(len(df), "items in dataset")
        return df, ["instruction"]
    
    elif name == "cognitivecomputations/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split":
        ds = load_dataset("cognitivecomputations/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split", split="train")
        rows = []
        for index, item in enumerate(ds):
            convs = item["conversations"]
            for turn, conv in enumerate(convs):
                if conv.get("value"):
                    rows.append({"content": conv["value"], "turn": turn, "index": index, "from": conv.get("from"), "key": "value"})
                if conv.get("markdown"):
                    rows.append({"content": conv["markdown"]['answer'], "turn": turn, "index": index, "from": conv.get("from"), "key": "markdown"})
                    print(conv["markdown"])
                if conv.get("text"):
                    rows.append({"content": conv["text"], "turn": turn, "index": index, "from": conv.get("from"), "key": "text"})
                    print(conv["text"])

        print(rows[0])
        df = Dataset.from_list(rows)
        return df, ["content"]
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
    

    if "heegyu" in model_name:
        system_prompt = 'You are a Korean translator. Translate your English text into Korean'
    elif "Seagull" in model_name:
        system_prompt = "주어진 문장을 한국어로 번역하세요."
    else:
        system_prompt = "You are a Korean translator. Translate given English text into Korean text. When translating the source code, only comments should be translated.\n\n===English Text===\nHello!"

    skip_length = estimate_skip_length(output_file)
    if skip_length > 0:
        dataset = dataset.select(range(skip_length, len(dataset)))
        print(f"Skipping {skip_length} examples")

    fout = jsonlines.open(output_file, "a")

    for batch in tqdm(batched_iteration(dataset, batch_size=batch_size), total=len(dataset)//batch_size):
        for key in keys:
            if "Qwen2" in model_name:
                prompts = [system_prompt + x[key] for x in batch]
                responses = model.generate_batch(prompts, histories=None, gen_args=gen_args, generation_prefix="===Korean Text===\n안녕하세요!")
            else:
                prompts = ["Hello! " + x[key] for x in batch]
                histories = [[{"role": "system", "content": system_prompt}] for _ in range(len(batch))]
                responses = model.generate_batch(prompts, histories=histories, gen_args=gen_args, generation_prefix="안녕하세요!")

            for item, response in zip(batch, responses):
                item[f"{key}_ko"] = response
    
        for item, response in zip(batch, responses):
            item["generator"] = model_name
            item["dataset_name"] = dataset_name
            fout.write(item)

if __name__ == "__main__":
    fire.Fire(main)