
mkdir -p data/
python -m src.translate_prompt \
    --model kuotient/Seagull-13b-translation --dataset cognitivecomputations/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split \
    --max_length 2048 --prompt_length 1024 --batch_size 4