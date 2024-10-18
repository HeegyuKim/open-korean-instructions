# open-korean-instructions
Open Korean Instructions는 언어모델을 학습하기 위한 공개 한국어 instruction dataset들을 모아둔 저장소입니다. <br/>
이 외에도  번역하거나 GPT를 이용해서 생성한 다양한 데이터들이 존재합니다. 혹시 새로운 데이터가 있다면 PR로 알려주세요.

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
| [maywell/ko_Ultrafeedback_binarized](https://huggingface.co/datasets/maywell/ko_Ultrafeedback_binarized) | 62k (RM) | 싱글턴 | Synatra-7B-Translation 모델을 통해 Ultrafeedback_binarized를 번역하고 정제한 데이터셋입니다. |
| [MrBananaHuman/kor_ethical_question_answer](https://huggingface.co/datasets/MrBananaHuman/kor_ethical_question_answer) | 29.1k | 싱글턴 | RLHF 학습을 위한 AI 윤리적/비윤리적 질의-답변 데이터셋 |
| [HumanF-MarkrAI/WIKI_QA_Near_dedup](https://huggingface.co/datasets/HumanF-MarkrAI/WIKI_QA_Near_dedup) | 138k | 싱글턴 | maywell(Jeonghwan Park)께서 만드신 maywell/wikidata_QA 에서 deduplication한 QA 데이터 |
| [kaist-ai/Multilingual-CoT-Collection](https://huggingface.co/datasets/kaist-ai/Multilingual-CoT-Collection) | 77.2k | 싱글턴 | KAIST에서 공개한 다국어 CoT collection, 한국어 77.2k 포함 |
| [heegyu/PKU-SafeRLHF-ko](https://huggingface.co/datasets/heegyu/PKU-SafeRLHF-ko) | 164k(RM)| 싱글턴 | [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) 데이터 번역 중 |
| [heegyu/hh-rlhf-ko](https://huggingface.co/datasets/heegyu/hh-rlhf-ko) | 113k(RM) | 멀티턴 | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) 데이터 번역 중 |
| [heegyu/webgpt_comparisons_ko](https://huggingface.co/datasets/heegyu/webgpt_comparisons_ko) | 19.6k(RM) | 싱글턴 | [openai/webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)를 모델로 번역 |
| [heegyu/glaive-function-calling-v2-ko](https://huggingface.co/datasets/heegyu/glaive-function-calling-v2-ko) | 15.2k (Function Calling) | 멀티턴 | [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) 중에서 15.2k 개를 ChatGPT로 번역 |
| [squarelike/ko_medical_chat](https://huggingface.co/datasets/squarelike/ko_medical_chat) | 3.04k | 멀티턴 | [jwj7140/ko-medical-chat](https://github.com/jwj7140/ko-medical-chat) [MedText](https://huggingface.co/datasets/BI55/MedText)와 [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor) 데이터셋을 GPT3.5를 통해 한국어 대화로 변환 |
| [MarkrAI/KoCommercial-Dataset](https://huggingface.co/datasets/MarkrAI/KoCommercial-Dataset) | 1.44M | 싱글턴 | 상업적으로 이용 가능한 데이터셋들을 수집 및 가공하여 하나로 병합 |
| [maywell/koVast](https://huggingface.co/datasets/maywell/koVast) | 685K | 멀티턴 | 685K의 대규모 멀티턴 한국어 대화 |
| [SJ-Donald/orca-dpo-pairs-ko](https://huggingface.co/datasets/SJ-Donald/orca-dpo-pairs-ko) | 36K | 싱글턴 | [mncai/orca_dpo_pairs_ko](https://huggingface.co/datasets/mncai/orca_dpo_pairs_ko), [Ja-ck/Orca-DPO-Pairs-KO](https://huggingface.co/datasets/Ja-ck/Orca-DPO-Pairs-KO), [We-Want-GPU/Yi-Ko-DPO-Orca-DPO-Pairs](https://huggingface.co/datasets/We-Want-GPU/Yi-Ko-DPO-Orca-DPO-Pairs) 3개의 DPO 데이터셋 병합 후 중복 제거 |
| [lcw99/wikipedia-korean-20240501-1million-qna](https://huggingface.co/datasets/lcw99/wikipedia-korean-20240501-1million-qna) | 1M | 싱글턴QA | 한글 위키피디아를 백만개의 섹션으로 나누고 백만개의 q&a를 생성 |
| [nlp-with-deeplearning/Ko.WizardLM_evol_instruct_V2_196k](https://huggingface.co/datasets/nlp-with-deeplearning/Ko.WizardLM_evol_instruct_V2_196k) | 196k | 싱글턴 | 자체 구축한 번역기로 WizardLM/WizardLM_evol_instruct_V2_196k을 번역한 데이터셋 |
| [HAERAE-HUB/qarv-instruct-100k](https://huggingface.co/datasets/HAERAE-HUB/qarv-instruct-100k) | 100k | 싱글턴 | 한국에 대한 지식이 필요한 지시문-답변 쌍 (영어 포함) |
| [kuotient/orca-math-word-problems-193k-korean](https://huggingface.co/datasets/kuotient/orca-math-word-problems-193k-korean) | 193k | 싱글턴 | microsoft/orca-math-word-problems-200k 번역 |
| [kuotient/orca-math-korean-preference](https://huggingface.co/datasets/kuotient/orca-math-korean-preference) | 193k | 싱글턴(DPO) | 번역된 microsoft/orca-math-word-problems-200k를 이용해 만든 DPO 데이터셋 |
| [jojo0217/korean_safe_conversation](https://huggingface.co/datasets/jojo0217/korean_safe_conversation) | 26k | 싱글턴 | 성균관대 - VAIV COMPANY 산학협력을 위해 구축한 일상대화 데이터로, 자연스럽고 윤리적인 챗봇 구축을 위한 데이터셋 |
| [HAERAE-HUB/K2-Feedback](https://huggingface.co/datasets/HAERAE-HUB/K2-Feedback) | 100k | 싱글턴 | K^2-피드백은 한국어 모델에서 세분화된 평가 능력을 향상시키기 위해 만들어진 데이셋, [Feedback Collection](https://huggingface.co/datasets/kaist-ai/Feedback-Collection)을 기반으로 한국 문화와 언어학에 특화된 지시문을 통합합니다. (NOTE: 원래 [Prometheus](https://arxiv.org/abs/2310.08491) 모델 학습 용 데이터지만 5점 output만을 가져와서 학습에 활용할 수 있다) |
| [maywell/kiqu_samples](https://huggingface.co/datasets/maywell/kiqu_samples) | 24.9k | 싱글턴 | kiqu-70b 모델의 출력 샘플입니다. |
| [CarrotAI/ko-instruction-dataset](https://huggingface.co/datasets/CarrotAI/ko-instruction-dataset) | 7k | 싱글턴 | WizardLM-2-8x22B 모델을 사용하여 생성한 한국어로 이루어진 고품질 한국어 데이터셋, WizardLM: Empowering Large Language Models to Follow Complex Instructions에서 소개된 방법으로 생성 | 
| [HAERAE-HUB/HR-Instruct-Math-v0.1](https://huggingface.co/datasets/HAERAE-HUB/HR-Instruct-Math-v0.1) | 30k | 싱글턴 | 한국어 수학 instruction 데이터 (PoC 버전) |
| [iknow-lab/qarv-instruct-ko-mt](https://huggingface.co/datasets/iknow-lab/qarv-instruct-ko-mt) | 10K | 멀티턴 | [HAERAE-HUB/qarv-instruct-ko](https://huggingface.co/datasets/HAERAE-HUB/qarv-instruct-ko) 데이터 1만여개에 GPT-3.5-turbo를 이용해서 2턴 대화를 더 추가한 멀티턴 데이터 |
| [iknow-lab/ko-evol-writing-wiki](https://huggingface.co/datasets/iknow-lab/ko-evol-writing-wiki) | 30K | 싱글턴 | GPT-3.5-turbo를 이용해서 생성한 글쓰기 / 창의적 글쓰기 데이터 |
| [AIHub RLHF dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71748) | SFT(13K), RM(33K), PPO(33K) | 싱글턴 | RM 데이터는 지시문과 5개 답변에 대해 순위가 매겨져있음. PPO 데이터의 경우 지시문만 있으며 답변 없음. |

## 다른 컬렉션
| 컬렉션 | 설명 |
|---|---|
| [유준혁님의 번역 데이터](https://huggingface.co/collections/youjunhyeok/en-ko-translate-6703474b419fcb9e5d6a7852) | 영어 데이터셋을 한글로 번역한 데이터셋입니다. | 
| [유준혁님의 번역 데이터 2(Magpie)](https://huggingface.co/collections/youjunhyeok/magpie-ko-66cbc570a9891d5b43a170d9) | Magpie 데이터셋 한국어 번역본 (@nayohan님 번역 모델 사용) |
| [songys/huggingface_KoreanDataset](https://github.com/songys/huggingface_KoreanDataset) | 송영숙님의 2024년 10월 10일 기준 huggingface에 있는 한국어 데이터 세트 정리 |
| [나요한님의 번역 데이터](https://huggingface.co/collections/nayohan/translated-en-ko-dataset-6665023b1036d124ede5f81c) | Datasets translated from English to Korean using llama3-instrucTrans-enko-8b`` |
## 평가용 데이터셋
| 이름 | # | 타입 | 내용 |
|---|---|---|---|
| [HAERAE-HUB/KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) | 243k | MCQA | 45개 주제의 전문가 수준 한국어 성능 평가 벤치마크 |
| [HAETAE-project/HAE-RAE-BENCH](https://github.com/HAETAE-project/HAE-RAE-BENCH) | 1.5k | MCQA | HAE-RAE Bench는 언어 모델의 한국어 능력(어휘, 역사, 상식, 독해)을 평가하기 위해 제작된 벤치마크 데이터셋입니다. |
| [HAERAE-HUB/CSAT-QA](https://huggingface.co/datasets/HAERAE-HUB/CSAT-QA) | 0.9k | MCQA | 국어 수능문제 |
| [HAERAE-HUB/K2-Eval](https://huggingface.co/datasets/HAERAE-HUB/K2-Eval) | 90 | 생성 | 정확한 답변을 위해서는 한국어 문화에 대한 깊이 있는 지식이 필요한 90개의 사람이 작성한 지시문, 사람 혹은 GPT-4가 평가 |
| [sean0042/KorMedMCQA](https://huggingface.co/datasets/sean0042/KorMedMCQA) | < 1k | MCQA | 한국어 의료 QA 벤치마크 |
| [HAERAE-HUB/Korean-Human-Judgements](https://huggingface.co/datasets/HAERAE-HUB/Korean-Human-Judgements) | < 1k | Human Preference | 각각 질문, 답변 A, 답변 B와 사람의 선호 표시 |
| [HAERAE-HUB/KUDGE](https://huggingface.co/datasets/HAERAE-HUB/KUDGE) | 2.8k | Human Preference | 한국어 응답에 대한 메타평가 능력을 검사하기위한 5.6k한국어 human annotation |

## 평가 플랫폼
- [Ko Chatbot Arena Leaderboard](https://huggingface.co/spaces/instructkr/ko-chatbot-arena-leaderboard): 사람이 여러 챗봇의 결과를 비교해보고 그 승률과 ELO 점수를 보여주는 리더보드
- [instructkr/LogicKor-leaderboard](https://huggingface.co/spaces/instructkr/LogicKor-leaderboard): 한국어 언어모델 다분야 사고력 벤치마크
- [호랑이 LLM 리더보드](https://wandb.ai/wandb-korea/korean-llm-leaderboard/reports/-LLM---Vmlldzo3MzIyNDE2?accessToken=95bffmg3gwblgohulknz7go3h66k11uqn1l3ytjma1uj3w0l0dwh1fywgsgpbdyy): wandb에서 공개한 Q&A, 멀티턴 형식의 한국어 LLM 평가 리더보드
- [ko-RM-judge](https://github.com/HeegyuKim/ko-rm-judge): 보상 모델(Reward Model)을 이용하여 챗봇의 대답을 평가하고, 그 점수를 비교
- [Korean-SAT-LLM-Leaderboard](https://github.com/minsing-jin/Korean-SAT-LLM-Leaderboard): 10년치 대한민국 수능시험 평가
- [KoMT-Bench](https://huggingface.co/datasets/LGAI-EXAONE/KoMT-Bench): MT벤치 한국어

## 한국어 합성 데이터 구축에 참고할 저장소
- [iKnowLab-Projects/ko-genstruct](https://github.com/iKnowLab-Projects/ko-genstruct)
- [lcw99/evolve-instruct](https://github.com/lcw99/evolve-instruct)



