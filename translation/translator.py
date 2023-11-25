from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline



SYNATRA_PROMPT = """<|im_start|>system
주어진 문장을 한국어로 번역해라.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

class Synatra:

    def __init__(self, device) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("maywell/Synatra-7B-v0.3-Translation")
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained("maywell/Synatra-7B-v0.3-Translation").eval().half().to(device)
        self.device = device
        
    def translate(self, prompts, **kwargs):        
        prompts = [SYNATRA_PROMPT.format(instruction=x) for x in prompts]

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        outs = self.model.generate(**inputs, **kwargs, early_stopping=True)
        return self.tokenizer.batch_decode(outs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
class NLLB:

    def __init__(self, device) -> None:
        model_id = "facebook/nllb-200-distilled-600M"
        #"NHNDQ/nllb-finetuned-en2ko"
        self.translator = pipeline('translation', model_id, device=device, src_lang='eng_Latn', tgt_lang='kor_Hang', max_length=512)

        
    def translate(self, texts, **kwargs):        
        outs = self.translator(texts, **kwargs, early_stopping=True)
        return [x["translation_text"] for x in outs]
    
  