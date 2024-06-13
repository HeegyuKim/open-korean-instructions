
mkdir -p data/
# python -m src.generate_response \
#     --model yanolja/EEVE-Korean-Instruct-2.8B-v1.0 --chat_template eeve
# python -m src.generate_response \
#     --model allganize/Llama-3-Alpha-Ko-8B-Instruct --eos_token '<|eot_id|>'

python -m src.generate_response \
    --model 42dot/42dot_LLM-SFT-1.3B --chat_template '42dot'

# python -m src.generate_response \
#     --model heegyu/beomi__gemma-ko-2b-gemma-ko-2b-0416@steps-200000 --eos_token "<end_of_turn>" --chat_template gemma

    