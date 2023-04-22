from datasets import load_dataset
import os
import json
import click
from itertools import chain
from copy import deepcopy
from typing import List, Any, Dict, Optional
from tqdm.auto import tqdm
import jsonlines
import re
import openai
import ray

# 너무 긴 context는 제거한다
MAX_LENGTH_TO_SKIP = 6120

class DataSource:
    def __getitem__(self, index):
        """
            return item: Dict[title, text]
        """
        pass

class KorQuAD(DataSource):

    def __init__(self) -> None:
        super().__init__()
        df = load_dataset("KETI-AIR/korquad", "v1.0", split="train").to_pandas()
        self.dataset = df[["title", "context"]].drop_duplicates()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset.iloc[index, :]
        return {
            'title': item['title'],
            'text': item['context']
        }
    
DATA_SOURCES = {
    "korquad-v1": KorQuAD
}


class Generator:
    def generate_dialogue(self, items: List[Dict]):
        pass

class ChatGPTGenerator(Generator):

    def __init__(self) -> None:
        super().__init__()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def _generate(self, item):
        prompt = """
        A와 B 두 사람이 나누는 대화를 만들어주세요. 
        1. A는 주어진 글에서 말하는 내용에 관해 B에게 질문을 하거나 요청합니다. B는 글의 내용을 이해하고 완전히 학습한 상태입니다. B는 A의 질문과 요청에 자신이 학습한 내용을 바탕으로 대답을 해야 합니다. 
        2. B는 글에 존재하지 않거나 사실에 근거하지 않은 대답을 해서는 안됩니다. 
        3. 각 발화는 최대 3개의 문장으로 이루어져 있습니다.
        4. 대화는 A와 B가 서로 주고받으며 순서대로 A의 발화는 A:, B의 발화는 B: 로 시작해야하고 띄어쓰기로 구분합니다.
        5. A와 B가 글을 읽었다는 내용이나, 글에 대해서 평가하거나, 글을 언급하는 내용이 포함되어서는 안됩니다.
        6. A와 B가 서로를 언급할 때는 A씨, B씨로 호칭해야합니다.
        7. A와 B는 서로 8번 대화를 주고받아야 합니다. 대화의 전체길이는 최대한 200 단어가 넘지 않도록 대화를 끝내야합니다.
        """.strip()

        query = f"제목: {item['title']}\n{item['text']}"

        if len(query) >= MAX_LENGTH_TO_SKIP:
            return None

        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ]
        )
        output_text = output["choices"][0]['message']['content']
        return output_text

    def generate_dialogue(self, items):
        for x in items:
            x["dialogue"] = self._generate(x)

        return items

def iter_batch(source: DataSource, start_index: int, batch_size: int, max_items: Optional[int]):
    items = []
    
    if max_items:
        end_index = start_index + min(len(source), max_items)
    else:
        end_index = start_index + len(source)

    for i in range(start_index, end_index):
        item = source[i]
        item["index"] = i

        items.append(item)

        if (i + 1) % batch_size == 0:
            yield items 
            items = []

    if items:
        yield items # generator.generate_dialogue(items)
    

@ray.remote
def generate(generator: Generator, batch: List[Dict]):
    return generator.generate_dialogue(batch)


@click.command()
@click.argument("dataset")
@click.argument("output")
@click.option("--batch-size", default=1, type=int)
@click.option("--num_process", default=1, type=int)
@click.option("--max-items", default=100, type=int)
@click.option("--resume", default=True, type=bool)
def main(
    dataset: str,
    output: str,
    num_process: int,
    resume: bool,
    max_items: Optional[int],
    batch_size: int
):
    ray.init(num_cpus=num_process)

    source = DATA_SOURCES[dataset]()
    print("total dataset", len(source))

    generator = ChatGPTGenerator()
    start_index = 0

    if os.path.exists(output):
        # 이전에 몇개까지 생성을 했었는지 확인
        with jsonlines.open(output) as f:
            for start_index, _ in enumerate(f):
                pass
            if start_index > 0:
                start_index += 1

        output_file = jsonlines.open(output, "a")
    else:
        output_file = jsonlines.open(output, "w")

    print("already processed items", start_index)

    iter_batches = iter(iter_batch(source, start_index, batch_size, max_items))
    progress = tqdm(total=len(source) // batch_size)
    progress.update(start_index)

    while True:
        batches = []
        try:
            for _ in range(num_process):
                batches.append(next(iter_batches))
        except StopIteration:
            pass

        if len(batches) == 0:
            break

        futures = [generate.remote(generator, b) for b in batches]
        for item in ray.get(futures):
            output_file.write(item)
        progress.update(num_process)

if __name__ == "__main__":
    main()