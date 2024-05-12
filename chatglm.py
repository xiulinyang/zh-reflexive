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
animacy_pro = Path('data/inanimate_pron.txt').read_text().strip().split('\n')
animacy_noun = Path('data/inanimate_nouns.txt').read_text().strip().split('\n')
subj_f1 = Path('data/subject_orientation_f1.txt').read_text().strip().split('\n')
subj_m1 = Path('data/subject_orientation_m1.txt').read_text().strip().split('\n')

def get_probability(zh_sents, output, blocking = False, female_first=False, animacy=False):
# Get logits from the model
    c=0
    f = []
    m = []
    w = []
    t = []
    with open(output, 'w') as out_tsv:
        out_tsv.write('he\ther\tme\tit\n')
        for sent in zh_sents:
            sent = f'“{sent}”，'
            sent+='在这句话中，自己指的是'
            encoded_input = tokenizer(sent, return_tensors='pt').to(model.device)
            token_ids = encoded_input['input_ids']

            with torch.no_grad():
                outputs = model(**encoded_input)
                logits = outputs.logits  # Assuming the model outputs include logits

            next_word_m = '他'
            next_word_f = '她'
            next_word_w = '我'
            next_word_t = '它'

            next_word_id_m = tokenizer.encode(next_word_m, add_special_tokens=False)[0]
            next_word_id_f = tokenizer.encode(next_word_f, add_special_tokens=False)[0]
            next_word_id_w = tokenizer.encode(next_word_w, add_special_tokens=False)[0]
            next_word_id_t = tokenizer.encode(next_word_t, add_special_tokens=False)[0]

            # Apply softmax to convert logits to probabilities
            softmax_probs = F.softmax(logits, dim=-1)

            # Extract the probability of "Monday" for the next word prediction
            next_word_probability_him = softmax_probs[0, -1, next_word_id_m].item()
            next_word_probability_her = softmax_probs[0, -1, next_word_id_f].item()
            next_word_probability_w = softmax_probs[0, -1, next_word_id_w].item()
            next_word_probability_t = softmax_probs[0, -1, next_word_id_t].item()



            f.append(next_word_probability_her)
            m.append(next_word_probability_him)
            w.append(next_word_probability_w)
            t.append(next_word_probability_t)



            all_prob = {'f': next_word_probability_her, 'm': next_word_probability_him,
                        'w': next_word_probability_w, 't': next_word_probability_t}
            out_tsv.write(f'{next_word_probability_him}\t{next_word_probability_her}\t{next_word_probability_w}\t{next_word_probability_t}\n')
            all_prob = sorted(all_prob.items(), key= lambda x:x[1], reverse=True)
            if blocking:
                if all_prob[0][0] =='w':
                    c += 1
            elif animacy:
                if all_prob[0][0] =='t':
                    c+=1
            else:
                if female_first:
                   if all_prob[0][0]=='m':

                        c+=1
                elif not female_first:
                    if all_prob[0][0]=='f':
                        c+=1
        print(c/len(zh_sents))
        print(c)
        print(len(zh_sents))
if __name__ == '__main__':
    print('In ambiguous setting, the percentage of local binding:')
    get_probability(amb_f1, 'amb_f1_glm.tsv', female_first=True)
    get_probability(amb_m1, 'amb_m1_glm.tsv', female_first=False)
    print('In externally oriented verb setting, the percentage of local binding:')
    get_probability(verb_f1, 'verb_f1_glm.tsv', female_first=True)
    get_probability(verb_m1, 'verb_m1_glm.tsv', female_first=False)
    print('In the blocking effect setting, the percentage of local binding:')
    get_probability(blocking, 'blocking_glm.tsv', blocking=True)
    print('In animate (pro) setting, the percentage of local binding:')
    get_probability(animacy_pro, 'animacy_pro_glm.tsv', animacy=True)
    print('In animate (noun) setting, the percentage of local binding:')
    get_probability(animacy_noun, 'animacy_noun_glm.tsv', animacy=True)

    print('In subject orientation, the percentage of local binding:')
    get_probability(subj_f1, 'subj_f1_glm.tsv', female_first=True)
    get_probability(subj_m1, 'subj_m1_glm.tsv', female_first=False)

