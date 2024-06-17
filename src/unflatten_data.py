import fire
import jsonlines
from tqdm import tqdm


def main(input: str, output: str):
    assert input != output
    lines = list(jsonlines.open(input))

    with jsonlines.open(output, "w") as fout:
        index = 0
        convs_ko, convs_en = [], []
        for line in tqdm(lines):
            if line['index'] != index:
                fout.write({
                    "messages": convs_en,
                    "messages_ko": convs_ko
                    })
                convs_ko, convs_en = [], []
                index = line['index']
                
            convs_en.append({
                "role": line["from"],
                "content": line["content"]
            })
            convs_ko.append({
                "role": line["from"],
                "content": line["content_ko"]
            })
    print(index)

if __name__ == "__main__":
    fire.Fire(main)