from datasets import load_dataset
from translator import Synatra
import jsonlines
from tqdm.auto import tqdm
import os


translator = Synatra("cuda:1")
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")

args = dict(
    max_new_tokens=512,
)
def translate_item(item):
    output = dict(
        prompt_k=translator.translate([item["prompt"].strip()], **args),
        response_0_ko=translator.translate([item["response_0"].strip()], **args),
        response_1_ko=translator.translate([item["response_1"].strip()], **args),
        **item
    )
    return output

os.makedirs("PKU-SafeRLHF-KO", exist_ok=True)

for split, ds in dataset.items():
    fout = jsonlines.open(f"./PKU-SafeRLHF-KO/{split}.json", "w")

    for item in tqdm(ds, desc=split):
        out = translate_item(item)
        fout.write(out)

    fout.close()

