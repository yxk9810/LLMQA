import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from transformers import  BitsAndBytesConfig

# # # Quantization
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float16)
checkpoint = 'baichuan-inc/Baichuan2-7B-Chat'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto",  trust_remote_code=True, quantization_config=q_config).eval()
model.generation_config = GenerationConfig.from_pretrained(checkpoint)
model.generation_config.num_beams=1
model.generation_config.top_k=1
import json

from tqdm import tqdm
test_data = json.load(open('test_rerank_1024_top5.json', 'r', encoding='utf-8'))
result = []
for td, d in enumerate(tqdm(test_data)):
    messages = []
    #   prompt = '请根据背景回答问题，如果背景中没有答案，请回答"无法回答"。'+'问题:'+d['question']+' 背景:'+''.join(d['contents'])
    prompt = '请根据背景内容回答问题"。' + '问题:' + d['question'] + ' 背景:' + ''.join(d['contents'])
    #   response, history = model.chat(tokenizer, prompt, history=None)
    messages.append({"role": "user", "content": prompt})
    response = model.chat(tokenizer, messages)
    json_data = {'question': d['question'], 'answer_1': '', 'answer_2': d['contents'][0], 'answer_3': d['contents'][1]}
    if '无法回答' in response:
        json_data['answer_1'] = json_data['answer_2']
    else:
        json_data['answer_1'] = response
    result.append(json_data)

import json
json.dump(result,open('baichuan_7b_large_result_1106.json','w',encoding='utf-8'),ensure_ascii=False)