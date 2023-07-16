# open-korean-instructions
Open Korean Instructions는 언어모델을 학습하기 위한 공개 한국어 instruction dataset들을 모아둔 데이터셋입니다. KoAlpaca v1.0과 v1.1, ShareGPT DeepL 번역, OIG-smallchip2-ko, KorQuAD-chat 5가지를 포멧을 통일하고 합쳤습니다. [Huggingface Link](https://huggingface.co/datasets/heegyu/open-korean-instructions). 

[Gradio Demo](https://huggingface.co/spaces/heegyu/gorani-v0) 위 데이터로 학습한 355M 모델을 사용해보실 수 있습니다.

이 외에도  번역하거나 GPT를 이용해서 생성한 다양한 데이터들이 존재합니다.

## 공개된 데이터 정리
| 이름 | # | 타입 | 내용 |
|---|---|---|---|
| [KoAlpaca v1.0](https://huggingface.co/datasets/Bingsu/ko_alpaca_data) | 52K | 싱글턴 | Alpaca instruction 번역 후 output을 ChatGPT로 생성 |
| [KoAlpaca v1.1](https://raw.githubusercontent.com/Beomi/KoAlpaca/main/KoAlpaca_v1.1.jsonl) | 21K | 싱글턴 | 지식인 질문 수집 후 ChatGPT로 대답 생성 |
| [ShareGPT DeepL 번역](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko) | 620K(싱글턴)<br/> 84K(멀티턴) | 멀티턴, 싱글턴 | ShareGPT 데이터를 DeepL로 번역 |
| [ShareGPT-74k-ko](https://huggingface.co/datasets/dbdu/ShareGPT-74k-ko) | 74k, 55k(코드제거) | 멀티턴 | ShareGPT 90k의 cleaned 버전을 구글 번역기를 이용하여 번역 |
| [KoChatGPT 실습](https://github.com/airobotlab/KoChatGPT) | 13K | 싱글턴, 멀티턴, RM | 한국어 질문 데이터셋에서 질문 수집 후 ChatGPT로 대답 생성 |
| [OIG-small-chip2-ko](https://huggingface.co/datasets/heegyu/OIG-small-chip2-ko) | 210K | 싱글턴 | LAION AI의 [OIG-smallchip-2](https://github.com/LAION-AI/Open-Instruction-Generalist) 영어 데이터 Google Translate으로 번역 |
| [Korquad-Chat](https://huggingface.co/datasets/heegyu/korquad-chat-v1) | 9.6K | 멀티턴, 지식기반 | [KorQuAD v1](https://huggingface.co/datasets/KETI-AIR/korquad) 데이터의 context(뉴스, 위키백과의 문단)을 주고, 관련 내용의 대화를 ChatGPT로 생성 |
| [AIRC-KETI/kowow](https://github.com/AIRC-KETI/kowow) | ? | 멀티턴, 지식기반 | WoW(Wizard Of Wikipedia) - 지식기반 대화 데이터를 번역한 데이터 |
| [CounselGPT](https://github.com/MrBananaHuman/CounselGPT) | 싱글턴(13k)<br/>멀티턴(8.7k) | 멀티턴, 싱글턴 | ChatGPT로 생성한 상담 데이터 |
| [Evolve-instruct](https://github.com/lcw99/evolve-instruct/) | 37k | 싱글턴 | [WizardLM]()에서 사용된 evol-instruct를 이용하여 instruction을 증강한 후 GP로 답변 생성한 데이터 |
| [KULLM v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2) | 153k | 싱글턴 | GPT4ALL, Dolly, Vicuna(ShareGPT) 데이터를 DeepL로 번역 |
| [psymon/namuwiki_alpaca_dataset](https://huggingface.co/datasets/psymon/namuwiki_alpaca_dataset/) | 79K | 싱글턴 | 나무위키 덤프 파일을 Stanford Alpaca 학습에 맞게 수정한 데이터셋 |
| [ko-lima-vicuna](https://huggingface.co/datasets/changpt/ko-lima-vicuna) | 1k | 싱글턴, 멀티턴(극히 일부) | GPT4 API를 사용하여 lima_vicuna_format 데이터를 한국어로 재생성한 데이터셋 |
| [Ko-StrategyQA](https://huggingface.co/datasets/PCEO-AI-CLUB/Ko-StrategyQA) | 2.2k(질문), 9k (문서) | Multi-hop QA, 예/아니오 단답형 | 이 데이터셋은 [StrategyQA](https://allenai.org/data/strategyqa)의 한국어 버전입니다. 기존 데이터셋의 모든 질문과 단락들을 DeepL을 사용하여 번역. |

### 그 외 instruction은 아니지만..
- [smilegate-ai/HuLiC](https://github.com/smilegate-ai/HuLiC)
- [smilegate-ai/OPELA](https://github.com/smilegate-ai/OPELA)


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
