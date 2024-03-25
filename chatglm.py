
from typing import List,Tuple
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()



#logits_processor = LogitsProcessorList()
#logits_processor.append(InvalidScoreLogitsProcessor())
with torch.no_grad():
    for idx, item in enumerate(['你好','怎么学习？']):
        ids = tokenizer.encode(item)
        input_ids = torch.LongTensor([ids]).cuda()
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0,
            return_dict_in_generate=True,
            output_scores = True
        )
        
        gen_ids = out.sequences.tolist()[0][len(input_ids):]
        response = tokenizer.decode(gen_ids)
        # scores是Tuple，只包括新生成的token的logits 形状为new_token_num * vocab_size
        scores = out.scores
        out_text = tokenizer.decode(gen_ids)
        answer = out_text.replace(item, "").replace("\nEND", "").strip()
        output = answer
        score = scores
        
        print(f"### {idx+1}.Answer:\n", output, score, '\n\n')
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

