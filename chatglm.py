import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from pathlib import Path
# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model.eval()

amb_f1 = Path('data/amb_f1.txt').read_text().strip().split('\n')
amb_m1 = Path('data/amb_m1.txt').read_text().strip().split('\n')
verb_f1 = Path('data/verb_f1.txt').read_text().strip().split('\n')
verb_m1 = Path('data/verb_m1.txt').read_text().strip().split('\n')
blocking = Path('data/blocking_amb.txt').read_text().strip().split('\n')


def get_probability(zh_sents, female_first=True):
# Get logits from the model
    c=0
    f = []
    m = []
    for sent in zh_sents:
        sent+='在这句话中，自己指的是'
        print(sent)
        encoded_input = tokenizer(sent, return_tensors='pt').to(model.device)
        token_ids = encoded_input['input_ids']

        with torch.no_grad():
            outputs = model(**encoded_input)
            logits = outputs.logits  # Assuming the model outputs include logits


        next_word_m = "他"
        next_word_id_m = tokenizer.encode(next_word_m, add_special_tokens=False)[0]

        next_word_f ='她'
        next_word_id_f = tokenizer.encode(next_word_f, add_special_tokens=False)[0]

        # Apply softmax to convert logits to probabilities
        softmax_probs = F.softmax(logits, dim=-1)

        # Extract the probability of "Monday" for the next word prediction
        next_word_probability_him = softmax_probs[0, -1, next_word_id_m].item()
        next_word_probability_her = softmax_probs[0, -1, next_word_id_f].item()
        print(next_word_probability_her, next_word_probability_him)
        f.append(next_word_probability_her)
        m.append(next_word_probability_him)

        if female_first:
            if next_word_probability_him>next_word_probability_her:
                c+=1
        else:
            if next_word_probability_her>next_word_probability_him:
                c+=1
    print(c/len(zh_sents))
    print(f, m)
if __name__ == '__main__':
    print('In ambiguous setting, the percentage of long-distance binding:')
    get_probability(amb_f1, female_first=True)
    get_probability(amb_m1, female_first=False)
    print('In externally oriented verb setting, the percentage of long-distance binding:')
    get_probability(verb_f1, female_first=True)
    get_probability(verb_m1, female_first=False)