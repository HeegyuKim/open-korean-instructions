import click
import json, jsonlines
import pandas
from datasets import load_dataset
import os
from tqdm.auto import tqdm


@click.group()
# @click.option('--debug/--no-debug', default=False)
@click.option('--user', default="<usr>")
@click.option('--bot', default="<bot>")
@click.option('--sys', default="<sys>")
@click.pass_context
def cli(ctx, user, bot, sys):
    ctx.ensure_object(dict)
    ctx.obj['user'] = user
    ctx.obj['bot'] = bot
    ctx.obj['sys'] = sys

@cli.command()
@click.option("--output", default="OKI-data/koalpaca.json")
@click.pass_context
def koalpaca(ctx, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print("koalpaca", ctx.obj)
    user, bot, sys = ctx.obj["user"], ctx.obj["bot"], ctx.obj["sys"]

    koalpaca = load_dataset("Bingsu/ko_alpaca_data", split="train")
    fout = jsonlines.open(output, "w")

    for item in tqdm(koalpaca, desc="processing koalpaca-v1.0"):
        instruction, input, generation = item["instruction"].strip(), item["input"].strip(), item["output"].strip()

        if input:
            fout.write({
                "source": "koalpaca-v1.0",
                "text": f"{user} {instruction}\n{sys} {input}\n{bot} {generation}"
            })
        else:
            fout.write({
                "source": "koalpaca-v1.0",
                "text": f"{user} {instruction}\n{bot} {generation}"
            })

    with jsonlines.open("data/KoAlpaca_v1.1.jsonl.txt") as koalpaca:
        for item in tqdm(koalpaca, desc="processing koalpaca-v1.1"):
            instruction, generation = item["instruction"].strip(), item["output"].strip()
            fout.write({
                "source": "koalpaca-v1.1",
                "text": f"{user} {instruction}\n{bot} {generation}"
            })

    fout.close()
    
@cli.command("oig-smallchip2-ko")
@click.option("--output", default="OKI-data/OIG-smallchip2-ko.json")
@click.pass_context
def oig_smallchip2_ko(ctx, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print("koalpaca", ctx.obj)
    user, bot, sys = ctx.obj["user"], ctx.obj["bot"], ctx.obj["sys"]

    koalpaca = load_dataset("heegyu/OIG-small-chip2-ko", split="train")
    fout = jsonlines.open(output, "w")

    for item in tqdm(koalpaca, desc="processing OIG-smallchip2-ko"):
        instruction, generation = item["user_translated"].strip(), item["chip2_translated"].strip()

        if input:
            fout.write({
                "source": "OIG-smallchip2-ko",
                "text": f"{user} {instruction}\n{bot} {generation}"
            })
            
def join_conversation(conv, user, bot):
    lines = []
    for uttr in conv:
        fr = uttr["from"].strip()
        value = uttr['value'].strip()

        if fr in ["gpt", "chatgpt", "bard"]:
            speaker = bot
        elif fr in ["bing", "bard"]: # ignore bing result
            return None
        elif fr in ["human", "user"]:
            speaker = user
        else:
            raise ValueError(f"invalid uttrance {uttr}")
        lines.append(f"{speaker} {value}")

    return "\n".join(lines)

@cli.command("sharegpt-deepl-ko")
@click.option("--output", default="OKI-data/sharegpt_deepl_ko.json")
@click.pass_context
def sharegpt_deepl_ko(ctx, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    user, bot, sys = ctx.obj["user"], ctx.obj["bot"], ctx.obj["sys"]
    print("sharegpt_deepl_ko", ctx.obj)
    
    with open("data/sharegpt_deepl_ko.json", encoding="utf-8") as fin, \
        jsonlines.open(output, "w") as fout:
        items = json.load(fin)

        for item in tqdm(items):
            text = join_conversation(item["conversations"], user, bot)
            
            if text:
                if bot not in text:
                    print("no bot", text)
                    continue

                fout.write({
                    "source": "sharegpt_deepl_ko",
                    "text": text,
                })


    


if __name__ == "__main__":
    cli(obj={})
