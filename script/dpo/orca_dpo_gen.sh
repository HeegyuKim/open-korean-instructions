
mkdir -p data/
python -m src.generate_response \
    --model yanolja/EEVE-Korean-Instruct-10.8B-v1.0 --chat_template eeve --dataset ultrafeedback

# python -m src.generate_response \
#     --model google/gemma-1.1-2b-it --eos_token "<end_of_turn>"

# python -m src.generate_response \
#     --model heegyu/beomi__gemma-ko-2b-gemma-ko-2b-0416@steps-499999 --eos_token "<end_of_turn>" --chat_template gemma

