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


class DataSource:
    def __getitem__(self, index):
        """
            return item: Any, texts: list[str]
        """
        pass

    def receive_translated_text(self, item: Dict, translate_texts: List[str]):
        
        pass

class OIGSmallChip2(DataSource):

    def __init__(self) -> None:
        super().__init__()
        self.dataset = load_dataset("0-hero/OIG-small-chip2", split="train")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        text1 = item["user"]
        text2 = item["chip2"]

        return item, [text1, text2]
    
    def receive_translated_text(self, item: Dict, translate_texts: List[str]):
        item["user_translated"] = translate_texts[0]
        item["chip2_translated"] = translate_texts[1]
        return item
    
DATA_SOURCES = {
    "oig-smallchip2": OIGSmallChip2
}

class Translator:

    def translate_texts(self, texts):
        return texts

class GoogleTranslator(Translator):

    def __init__(self) -> None:
        super().__init__()
        # Imports the Google Cloud Translation library
        from google.cloud import translate

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-service-key.json"
        with open("google-service-key.json") as f:
            project_id = json.load(f)["project_id"]

        self.client = translate.TranslationServiceClient()
        location = "global"
        self.parent = f"projects/{project_id}/locations/{location}"


    # Initialize Translation client
    def translate_texts(self, texts):
        """Translating Text."""
        # return texts # for test

        response = self.client.translate_text(
            request={
                "parent": self.parent,
                "contents": texts,
                "mime_type": "text/plain",
                "source_language_code": "en-US",
                "target_language_code": "ko",
            }
        )

        return [t.translated_text for t in response.translations]

class ChatGPTTranslator(Translator):

    def __init__(self) -> None:
        super().__init__()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def translate_texts(self, texts):
        
        prompt = """
        Translate these English texts to Korean. 
        """
        texts = [f"#{i}: {t}" for i, t in enumerate(texts)]

        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "\n".join(texts)},
                ]
        )
        output_text = output["choices"][0]['message']['content']
        texts = re.split(r"#\d+:", output_text)
        texts = [t for t in texts if t]
        # print(output_text)
        # print(texts)
        return texts
    
def batch_translate(translator, datasource: DataSource, items: List[Dict], texts: List[List[str]]):
    all_text = list(chain(*texts))
    translated_texts = translator.translate_texts(all_text)

    new_items = []
    i = 0

    for item, T in zip(items, texts):
        item = deepcopy(item)
        translations = translated_texts[i:i + len(T)]
        i += len(T)
        
        new_items.append(datasource.receive_translated_text(item, translations))

    return new_items

def iter_batch(translator, source: DataSource, start_index: int, batch_size: int, max_items: Optional[int]):
    items = []
    texts = []

    
    if max_items:
        end_index = start_index + min(len(source), max_items)
    else:
        end_index = start_index + len(source)

    for i in range(start_index, end_index):
        item, text = source[i]
        item["index"] = i

        items.append(item)
        texts.append(text)

        if (i + 1) % batch_size == 0:
            yield from batch_translate(translator, source, items, texts)
            items = []
            texts = []

    if items:
        yield from batch_translate(translator, source, items, texts)
    


@click.command()
@click.argument("dataset")
@click.argument("output")
@click.option("--batch-size", default=8, type=int)
@click.option("--max-items", default=100, type=int)
@click.option("--resume", default=True, type=bool)
def main(
    dataset: str,
    output: str,
    resume: bool,
    max_items: Optional[int],
    batch_size: int
):
    # translator = GoogleTranslator()
    translator = ChatGPTTranslator()
    source = DATA_SOURCES[dataset]()
    start_index = 0

    if os.path.exists(output):
        # 이전에 몇개까지 번역을 했었는지 확인
        with jsonlines.open(output) as f:
            for start_index, _ in enumerate(f):
                pass
            if start_index > 0:
                start_index += 1

        output_file = jsonlines.open(output, "a")
    else:
        output_file = jsonlines.open(output, "w")

    progress = tqdm(iter_batch(translator, source, start_index, batch_size, max_items))
    for item in progress:
        output_file.write(item)

if __name__ == "__main__":
    # texts = ["I've been having trouble sleeping. What could it be?",
    # "Who discovered the ‘Charles Denison equine encephalitis virus’?",
    # "I'm having trouble with my car. What should I do?",
    # "What is the best way to get my child to like reading?",
    # "How long do I have to file a lawsuit?"]
    # translator = ChatGPTTranslator()
    # print(translator.translate_texts(texts))
    main()