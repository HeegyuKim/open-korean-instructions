import openai, os, re
from datasets import load_dataset
import time
from copy import deepcopy
from pprint import pprint
import jsonlines
import json
import retry


class ChatGPTTranslator:

    def __init__(self) -> None:
        super().__init__()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @retry.retry(tries=3, delay=30)
    def translate_texts(self, convs):
        
        prompt = """
        You are a Korean translator. Data in the format of a given json array contains conversations between user and assistant. Each element in the array has roles and contents. You must translate the content value of the element when the role is user or assistant. You must also meet the following conditions.
        1. The result must be preserved in json format.
        2. The tone of the translated text should be a natural everyday conversation tone. 
        3. The translation content should not include the content that you are translating."""

        text = json.dumps(convs, ensure_ascii=False, indent=1)
        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ]
        )
        output_text = output["choices"][0]['message']['content']
        print(output_text)
        return output_text
    


def main():
    ds = load_dataset("heegyu/glaive-function-calling-v2-formatted", split="train")
    translator = ChatGPTTranslator()

    target_file = "../data/glaive-function-calling-v2-ko.json"
    if os.path.exists(target_file):
        with jsonlines.open(target_file) as fin:
            skip = len(list(fin))
        fout = jsonlines.open(target_file, "a")
    else:
        skip = 0
        fout = jsonlines.open(target_file, "w")

    print(f"skip {skip} items")
    for i, item in enumerate(ds):
        if i < skip:
            continue

        new_convs = []
        # for conv in item["conversations"]:
        #     role = conv["role"]
        #     conv = deepcopy(conv)
        #     if role in ["user", "assistant"]:
        #         content_ko = translator.translate_texts(role, conv["content"])
        #         conv = deepcopy(conv)
        #         conv["content"] = content_ko
        #     new_convs.append(conv)

        new_convs = translator.translate_texts(item["conversations"])
        # print(new_convs)
        time.sleep(2.0)

        item["conversations-ko"] = new_convs
        fout.write(item)

        if i == 20000:
            break
    
    fout.close()


if __name__ == "__main__":
    main()