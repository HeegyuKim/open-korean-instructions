"""

evol-instruct
koalpaca v1.1
kullm v2 - dolly, gpt4all 2개만
changpt/ko-lima-vicuna
HAERAE-HUB/KoInstruct-QA
heegyu/korquad-chat-v1
heegyu/kowow
NomaDamas/Ko-StrategyQA
"""
import click
import os
import json, jsonlines
from tqdm import tqdm
import requests
import pandas as pd
import random
from datasets import load_dataset
from traceback import print_exc
from pprint import pprint
from itertools import chain


def download(url, filename, read=False, encoding="utf-8"):
    os.makedirs("download", exist_ok=True)

    filename = os.path.join("download", filename)

    if not os.path.exists(filename):
        with open(filename, "w", encoding=encoding) as f:
            print(f"download {url} -> {filename}")
            text = requests.get(url).text
            f.write(text)
    elif read:
        with open(filename, encoding="utf-8") as f:
            text = f.read()
    else:
        return None
    
    return text

def load_jsonl_text(text):
    return [json.loads(x) for x in text.split("\n") if x]

HUMAN_FROMS = ["user"]
BOT_FROMS = ["gpt", "bard"]

def write_to_file(iterable, item_handler, filename):

    fout = jsonlines.open(f"data/{filename}.json", "w")

    for item in tqdm(iterable, desc=filename):
        try:
                
            if item_handler is not None:
                conv = item_handler(item)
            else:
                conv = item["conversations"]
            
            if conv is None:
                continue

            for c in conv:
                if c["from"] in HUMAN_FROMS:
                    c["from"] = "human"
                elif c["from"] in BOT_FROMS:
                    c["from"] = "bot"

            item = {
                "conversations": conv
            }
            fout.write(item)
        except:
            pprint(conv)
            print_exc()
            raise KeyboardInterrupt()

    fout.close()


def process_alpaca_format(dataset, output_name, inst_key="instruction", output_key="output", input_key="input"):
    def handler(item):
        output = [
            {
                "from": "human",
                "value": item[inst_key]
            },
            {
                "from": "bot",
                "value": item[output_key]
            }
        ]
        item_input = item.get(input_key, "")
        if item_input and item_input != "nan":
            output.insert(0, {
                "from": "input",
                "value": item_input
            })

        return output

    write_to_file(dataset, handler, output_name)

@click.group()
def cli():
    pass

@cli.command()
def strategyqa():
    url_items = "https://huggingface.co/datasets/NomaDamas/Ko-StrategyQA/raw/main/ko-strategy-qa_full.json"
    # url_para = "https://huggingface.co/datasets/NomaDamas/Ko-StrategyQA/resolve/main/ko-strategy-qa_paragraphs.csv"

    items = json.loads(download(url_items, "ko-strategy-qa_full.json", read=True))
    # download(url_para, "ko-strategy-qa_paragraphs.csv", encoding="cp949")

    # df = pd.read_csv("download/ko-strategy-qa_paragraphs.csv")
    # print(df)

    # pprint(list(items.values())[:10])
    
    response = {
        "True": ["그렇습니다", "예", "네", "맞아요", "예 맞습니다"],
        "False": ["아닙니다", "아니요", "아닙니다", "그렇지 않아요"]
    }

    def handler(item):
        res = random.choice(response[item["answer"]])
        output = [
            {
                "from": "human",
                "value": item["question"]
            },
            {
                "from": "bot",
                "value": res + " " + " ".join(item["facts"])
            }
        ]

        return output

    write_to_file(items.values(), handler, "strategy_qa")
    
@cli.command()
def korquadchat():
    dataset = load_dataset("heegyu/korquad-chat-v1", split="train")

    def conv_mapper(conv):
        
        if conv.startswith("<usr>") or conv.startswith("A :") or conv.startswith("A:"):
            who = "human"
            text = conv.replace("<usr>", "").replace("A :", "").replace("A:", "").strip()
        elif conv.startswith("<sys>"):
            who = "input"
            text = conv.replace("<sys>", "").strip()
        elif conv.startswith("<bot>") or conv.startswith("B:") or conv.startswith("B :"):
            who = "bot"
            text = conv.replace("<bot>", "").replace("B:", "").replace("B :", "").strip()
        else:
            raise ValueError(f"이상한거 발견 '{conv}'")

        return {
            "from": who,
            "value": text
        }
    
    def handler(item):
        if "\nC:" in item["text"]:
            return None
        convs = item["text"].strip().split("\n")
        try:
            convs = [conv_mapper(x.strip()) for x in convs if x.strip()]
        except ValueError as e:
            print_exc()
            print(item)
            return None

        return convs

    write_to_file(dataset, handler, "korquadchat")

@cli.command()
def koinstruct_qa():
    dataset = load_dataset("HAERAE-HUB/KoInstruct-QA", split="train")
    process_alpaca_format(dataset, "koinstruct_qa")

@cli.command()
def kowow():
    # dataset = load_dataset("heegyu/kowow", split="train")
    url = "https://huggingface.co/datasets/heegyu/kowow/resolve/main/train_ko.json"
    dataset = json.loads(download(url, "kowow.json", True))

    speaker2from = {
        "0_Wizard": "bot",
        "1_Wizard": "bot",
        "1_Apprentice": "human",
        "0_Apprentice": "human"
    }

    def parse_item(item):
        if item["text"] == "no_passages_used":
            return None
        
        output = [{
            "from": speaker2from[item["speaker"]],
            "value": item["text"]
        }]
        
        if "checked_sentence" in item and len(item["checked_sentence"]) > 0:
            output.insert(0, {
                "from": "input",
                "value": list(item["checked_sentence"].values())[0]
            })

        return output
    
    def handler(item):
        convs = item["dialog"]
        convs = map(parse_item, convs)
        convs = filter(lambda x: x is not None, convs)
        # print(list(convs))
        convs = chain(*convs)
        convs = list(convs)

        for i in range(1, len(convs)):
            if convs[i]['from'] == 'input':
                a = convs[i - 1]
                b = convs[i]
                convs[i - 1] = b
                convs[i] = a
                
        return convs
    
    write_to_file(dataset, handler, "kowow")


@cli.command()
def kullm_v2_dolly_gpt4all():
    dataset = load_dataset("nlpai-lab/kullm-v2", split="train")
    def filter_fn(item):
        if "dolly" in item["id"]:
            return True
        if "gpt4all" in item["id"]:
            return True
        else:
            return False
        
    dataset = dataset.filter(filter_fn)
    process_alpaca_format(dataset, "kullm_v2")

@cli.command()
def sharegpt_clean_nocode():
    # url = "https://huggingface.co/datasets/dbdu/ShareGPT-74k-ko/resolve/main/part2_ko_cleaned.json?download=true"
    dataset = load_dataset("dbdu/ShareGPT-74k-ko", data_files={"train": "part2_ko_cleaned.json"})["train"]
    write_to_file(dataset, None, "sharegpt_clean_nocode")

@cli.command()
def kolima():
    dataset = load_dataset("changpt/ko-lima-vicuna", split="train")
    write_to_file(dataset, None, "kolima")

@cli.command()
def evol_instruct():
    url = "https://raw.githubusercontent.com/lcw99/evolve-instruct/main/evol_instruct.json"
    def handler(item):
        return [
            {
                "from": "human",
                "value": item["input"]
            },
            {
                "from": "bot",
                "value": item["output"]
            },
        ]
    items = json.loads(requests.get(url).text)
    write_to_file(items, handler, "evol_instruct")

@cli.command()
def koalpaca_v1_1():
    
    def handler(item):
        return [
            {
                "from": "user",
                "value": item["instruction"]
            },
            {
                "from": "bot",
                "value": item["output"]
            },
        ]
    
    items = load_jsonl_text(requests.get("https://raw.githubusercontent.com/Beomi/KoAlpaca/main/KoAlpaca_v1.1.jsonl").text)
    write_to_file(items, handler, "koalpaca_v1.1")

@cli.command()
def kocot_2000():
    dataset = load_dataset("kyujinpy/KoCoT_2000", split="train")
    process_alpaca_format(dataset, "kocot_2000", inst_key="source", output_key="rationale")


@cli.command()
def openorca_ko():
    dataset = load_dataset("kyujinpy/OpenOrca-KO", split="train")
    process_alpaca_format(dataset, "openorca_ko")

@cli.command()
def wikiqa_dedup():
    dataset = load_dataset("HumanF-MarkrAI/WIKI_QA_Near_dedup", split="train")
    process_alpaca_format(dataset, "wikiqa_dedup")
    
@cli.command()
def kopen_platypus():
    dataset = load_dataset("kyujinpy/KOpen-platypus", split="train")
    process_alpaca_format(dataset, "kopen_platypus")

@cli.command()
def hrc():
    dataset = load_dataset("heegyu/HRC", split="train")
    process_alpaca_format(dataset, "hrc", inst_key="질의", output_key="상담례, 결정례 기반 답변")

@cli.command()
def kor_ethical_question_answer():
    dataset = load_dataset("MrBananaHuman/kor_ethical_question_answer", split="train")
    dataset = dataset.filter(lambda x: x["label"] == 0)
    process_alpaca_format(dataset, "kor_ethical_question_answer", inst_key="question", output_key="answer")
    
if __name__ == "__main__":
    cli()