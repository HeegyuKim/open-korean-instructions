# open-korean-instructions
Open Korean Instructions는 언어모델을 학습하기 위한 공개 한국어 instruction dataset들을 모아둔 저장소입니다. <br/>
이 외에도  번역하거나 GPT를 이용해서 생성한 다양한 데이터들이 존재합니다. 혹시 새로운 데이터가 있다면 PR로 알려주세요.

## 공개 데이터를 포멧을 통일하고 병합한 데이터
| 이름 | # | 데이터 |
|---|---|---|
| [open-korean-instructions](https://huggingface.co/datasets/heegyu/open-korean-instructions) | 376K | KoAlpaca v1.0과 v1.1, ShareGPT DeepL 번역, OIG-smallchip2-ko, KorQuAD-chat |
| [AULM-0809](https://huggingface.co/datasets/heegyu/aulm-0809) | 171K | KoAlpaca v1.1, ShareGPT-74k-ko의 코드제거 버전, KorQuAD-chat,  evolve-instruct, KoInstruct-QA, ko-lima-vicuna-kullm-v2의 GPT4ALL, Dolly 데이터 |

## 공개 데이터를 이용해서 직접 학습한 모델 모음
| 이름 | 크기 | 데이터 |
| --- | --- | --- |
| [heegyu/gorani-v0](https://huggingface.co/heegyu/gorani-v0) | 355M | [open-korean-instructions](https://huggingface.co/datasets/heegyu/open-korean-instructions) |
| [heegyu/polyglot-ko-1.3b-chat](https://huggingface.co/heegyu/polyglot-ko-1.3b-chat) | 1.3B | [AULM-0809](https://huggingface.co/datasets/heegyu/aulm-0809) |
| [heegyu/polyglot-ko-3.8b-chat](https://huggingface.co/heegyu/polyglot-ko-3.8b-chat) | 3.8B | [AULM-0809](https://huggingface.co/datasets/heegyu/aulm-0809) |
| [heegyu/KoLIMA-5.8b](https://huggingface.co/heegyu/KoLIMA-5.8b) | 5.8B | [changpt/ko-lima-vicuna](https://huggingface.co/datasets/changpt/ko-lima-vicuna) |
| [heegyu/polyglot-ko-5.8b-chat](https://huggingface.co/heegyu/polyglot-ko-5.8b-chat) | 5.8B | [AULM-0809](https://huggingface.co/datasets/heegyu/aulm-0809) |
| [heegyu/llama-2-ko-7b-chat](https://huggingface.co/heegyu/llama-2-ko-7b-chat) | 7B | [AULM-0809](https://huggingface.co/datasets/heegyu/aulm-0809) |
| [iknow-lab/AULM-12.8b-v0](https://huggingface.co/iknow-lab/AULM-12.8b-v0) | 12.8B | [AULM-0809](https://huggingface.co/datasets/heegyu/aulm-0809) |

[355M 모델 Gradio Demo](https://huggingface.co/spaces/heegyu/gorani-v0)

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
| [CounselGPT](https://github.com/MrBananaHuman/CounselGPT) | 싱글턴(13k)<br/>멀티턴(8.7k) | 멀티턴, 싱글턴 | GPT로 생성한 상담 데이터 |
| [Evolve-instruct](https://github.com/lcw99/evolve-instruct/) | 37k | 싱글턴 | [WizardLM]()에서 사용된 evol-instruct를 이용하여 instruction을 증강한 후 GP로 답변 생성한 데이터 |
| [KULLM v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2) | 153k | 싱글턴 | GPT4ALL, Dolly, Vicuna(ShareGPT) 데이터를 DeepL로 번역 |
| [nlpai-lab/openassistant-guanaco-ko](https://huggingface.co/datasets/nlpai-lab/openassistant-guanaco-ko) | 9.85k | 멀티턴 | Korean translation of Guanaco via the DeepL API |
| [psymon/namuwiki_alpaca_dataset](https://huggingface.co/datasets/psymon/namuwiki_alpaca_dataset/) | 79K | 싱글턴 | 나무위키 덤프 파일을 Stanford Alpaca 학습에 맞게 수정한 데이터셋 |
| [changpt/ko-lima-vicuna](https://huggingface.co/datasets/changpt/ko-lima-vicuna) | 1k | 싱글턴, 멀티턴(극히 일부) | GPT4 API를 사용하여 lima_vicuna_format 데이터를 한국어로 재생성한 데이터셋 |
| [taeshahn/ko-lima](https://huggingface.co/datasets/taeshahn/ko-lima) | 1k | 싱글턴, 멀티턴(극히 일부) | [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) (Zhou et al., 2023)의 [학습 데이터](https://huggingface.co/datasets/GAIR/lima)를 한국어로 번역한 데이터셋 |
| [Ko-StrategyQA](https://huggingface.co/datasets/PCEO-AI-CLUB/Ko-StrategyQA) | 2.2k(질문), 9k (문서) | Multi-hop QA, 예/아니오 단답형 | 이 데이터셋은 [StrategyQA](https://allenai.org/data/strategyqa)의 한국어 버전입니다. 기존 데이터셋의 모든 질문과 단락들을 DeepL을 사용하여 번역. |
| [HAERAE-HUB/KoInstruct-Base](https://huggingface.co/datasets/HAERAE-HUB/KoInstruct-Base) | 52k | 싱글턴 | Alpaca 데이터 번역인 듯 함. |
| [HAERAE-HUB/KoInstruct-QA](https://huggingface.co/datasets/HAERAE-HUB/KoInstruct-QA) | 50.3k | 싱글턴 | 원본 데이터가 뭔지 모르겠음. 위 데이터중에 중복이 있을 수도 있음. |
| [kyujinpy/KOpen-platypus](https://huggingface.co/datasets/kyujinpy/KOpen-platypus) | 24.9k | 싱글턴 | [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) 데이터 번역 |
| [ziozzang/EverythingLM-data-V2-Ko](https://huggingface.co/datasets/ziozzang/EverythingLM-data-V2-Ko) | 1k | 싱글턴 | [EverythingLM-data-V2](https://huggingface.co/datasets/totally-not-an-llm/EverythingLM-data-V2)를 DeepL로 번역 |
| [human-rights-corpus/HRC/](https://github.com/human-rights-corpus/HRC/) | 1.5k | 싱글턴 | 대화형 생성 모델을 위한 인권코퍼스 구축 - 대한민국 [국가인권위원회](https://case.humanrights.go.kr/dici/diciList.do)의 결정례와 상담사례 참조, 문체 변경과 질의 응답으로 변경하기 위해서 전후 맥락을 고려한 예시문을 만들고 GPT-3.5-turbo 을 이용하여 원샷 학습후 문답 생성 |
| [kyujinpy/OpenOrca-KO](https://huggingface.co/datasets/kyujinpy/OpenOrca-KO) | 21.6k | 싱글턴 | [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) dataset 중 약 2만개를 sampling하여 번역한 데이터셋 |
| [kyujinpy/KoCoT_2000](https://huggingface.co/datasets/kyujinpy/KoCoT_2000) | 2.16k | 싱글턴 | Using DeepL dataset, translation about [kaist-CoT](https://huggingface.co/datasets/kaist-ai/CoT-Collection). |
| [RLHF-Korean-Friendly-LLM](https://github.com/VAIV-2023/RLHF-Korean-Friendly-LLM) | 2.4K(SFT), 3.8K(RM), 3.6K(RLHF) | 싱글턴 | 다양한 데이터를 수집하여 RLHF를 위한 천개 단위의 데이터셋 구축 |
| [jojo0217/korean_rlhf_dataset](https://huggingface.co/datasets/jojo0217/korean_rlhf_dataset) | 107k | 싱글턴 | 성균관대학교 산학협력프로젝트 과정에서 한국어 llm 모델 SFT 학습을 위해 구축한 데이터셋 입니다. | 
| [maywell/ko_hh-rlhf-20k_filtered](https://huggingface.co/datasets/maywell/ko_hh-rlhf-20k_filtered) | 20k | 멀티턴, RM | hh-rlhf 데이터셋 중 20k를 synatra-translation 모델로 번역 |
| [squarelike/OpenOrca-gugugo-ko](https://huggingface.co/datasets/squarelike/OpenOrca-gugugo-ko) | 640k + (번역중) | 싱글턴 | [Gugugo-koen-7B-V1.1](https://huggingface.co/squarelike/Gugugo-koen-7B-V1.1)을 이용하여 [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)데이터셋을 번역 중 |

HAERAE-HUB 데이터셋들은 설명이 없어서 어떤 데이터인지 잘 모르겠네요.

## 평가용 데이터셋
| 이름 | # | 타입 | 내용 |
|---|---|---|---|
| [HAETAE-project/HAE-RAE-BENCH](https://github.com/HAETAE-project/HAE-RAE-BENCH) | 1.5k | ? | HAE-RAE Bench는 언어 모델의 한국어 능력(어휘, 역사, 상식, 독해)을 평가하기 위해 제작된 벤치마크 데이터셋입니다. |
| [HAERAE-HUB/CSAT-QA](https://huggingface.co/datasets/HAERAE-HUB/CSAT-QA) | 0.9k | 싱글턴, 객관식 | 국어 수능문제 |


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
