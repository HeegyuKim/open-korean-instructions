# open-korean-instructions
Open Korean Instructions는 언어모델을 학습하기 위한 공개 한국어 instruction dataset들을 모아둔 데이터셋입니다. KoAlpaca v1.0과 v1.1, ShareGPT DeepL 번역, OIG-smallchip2-ko, KorQuAD-chat 5가지를 포멧을 통일하고 합쳤습니다. [Huggingface Link](https://huggingface.co/datasets/heegyu/open-korean-instructions). 

[Gradio Demo](https://huggingface.co/spaces/heegyu/gorani-v0) 위 데이터로 학습한 355M 모델을 사용해보실 수 있습니다.

## 공개된 데이터 정리
| 이름 | # | 타입 |
|---|---|---|
| [KoAlpaca v1.0](https://huggingface.co/datasets/Bingsu/ko_alpaca_data) | 52K | 싱글턴 |
| [KoAlpaca v1.1](https://raw.githubusercontent.com/Beomi/KoAlpaca/main/KoAlpaca_v1.1.jsonl) | 21K | 싱글턴 |
| [ShareGPT DeepL 번역](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko) | 620K(싱글턴), 84K(멀티턴) | 멀티턴, 싱글턴 |
| [KoChatGPT 실습](https://github.com/airobotlab/KoChatGPT) | 13K | 싱글턴, 멀티턴, RM |
| [OIG-small-chip2-ko](https://huggingface.co/datasets/heegyu/OIG-small-chip2-ko) | 210K | 싱글턴 |
| [Korquad-Chat](https://huggingface.co/datasets/heegyu/korquad-chat-v1) | 9.6K | 멀티턴, 지식기반 |

## 그 외의 공개 대화 데이터(추후 전처리해서 추가할 예정)
- [AIRC-KETI/kowow](https://github.com/AIRC-KETI/kowow): WoW(Wizard Of Wikipedia) - 지식기반 대화 데이터를 번역한 데이터
- [smilegate-ai/HuLiC](https://github.com/smilegate-ai/HuLiC)
- [smilegate-ai/OPELA](https://github.com/smilegate-ai/OPELA)
- [CounselGPT](https://github.com/MrBananaHuman/CounselGPT) 

## 데이터 생성 코드
일부 데이터는 번역되거나 ChatGPT를 통해 생성했습니다.<br/>
`src/`에 있는 코드를 이용하여 데이터를 생성할 수 있습니다.

### Translate API를 이용하여 번역
```bash
python translate.py --max-items 10000 --batch-size 8 oig-smallchip2 ../data/oig-smallchip2.jsonl

# google은 비싸요 ㅠ. 기본 chatgpt
python translate.py --max-items 10000 --batch-size 8 --translator google oig-smallchip2 ../data/oig-smallchip2.jsonl
```

### ChatGPT로 지식기반대화 생성
```bash
python generate_kg_dialogue.py --max-items 10000 --batch-size 1 --num_process 4 korquad-v1 ../data/korquad-chat.jsonl
```

주의사항
- 서로를 A씨, B씨로 호칭합니다. 추후 전처리가 필요합니다.
- 할루시네이션이 있을 수 있습니다. 최대한 없애고자 주어진 정보 내에서만 대화하도록 프롬프트를 구성했습니다.
