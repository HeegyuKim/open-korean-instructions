
mkdir -p data/

python -m src.generate_response \
    --model google/gemini-1.5-flash --api_delay 15

    