import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from pathlib import Path
# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model.eval()

local_f1 = Path('data/local_female.txt').read_text().strip().split('\n')
local_m1 = Path('data/local_male.txt').read_text().strip().split('\n')
amb_f1 = Path('data/amb_f1.txt').read_text().strip().split('\n')
amb_m1 = Path('data/amb_m1.txt').read_text().strip().split('\n')
verb_f1 = Path('data/verb_f1.txt').read_text().strip().split('\n')
verb_m1 = Path('data/verb_m1.txt').read_text().strip().split('\n')
in_verb_f1 = Path('data/in_verb_f1.txt').read_text().strip().split('\n')
in_verb_m1 = Path('data/in_verb_m1.txt').read_text().strip().split('\n')
blocking = Path('data/blocking_amb.txt').read_text().strip().split('\n')
animacy_pro = Path('data/inanimate_pron.txt').read_text().strip().split('\n')
animacy_noun = Path('data/inanimate_nouns.txt').read_text().strip().split('\n')
subj_f1 = Path('data/subject_orientation_f1.txt').read_text().strip().split('\n')
subj_m1 = Path('data/subject_orientation_m1.txt').read_text().strip().split('\n')
subj_f1_bias = Path('data/subject_orientation_f1_bias.txt').read_text().strip().split('\n')
subj_m1_bias = Path('data/subject_orientation_m1_bias.txt').read_text().strip().split('\n')

natural_local_m = Path('data/filtered_sents_local_m_binding.txt').read_text().strip().split('\n')
natural_local_f = Path('data/filtered_sents_local_f_binding.txt').read_text().strip().split('\n')

natural_local_verb = Path('data/real_data_lb_name.txt').read_text().strip().split('\n')
natural_long_verb = Path('data/real_data_ldb_verb.txt').read_text().strip().split('\n')
natural_long_anim = Path('data/real_data_ldb_anim.txt').read_text().strip().split('\n')
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
            ziji_index = sent.index('自')
            sent = f'{sent[:ziji_index - 1]}{sent[ziji_index:-1]}'
            sent = f'在“{sent}”这句话中，自己指的是'
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


            softmax_probs = F.softmax(logits, dim=-1)


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
    return c, len(zh_sents)


def test_real_data(zh_sents, output):
    c = 0
    f = []
    m = []
    w = []
    t = []
    zh_sents = [x.split() for x in zh_sents]
    label2target = {'f': '她', 'm': '他', 'w': '我', 't': '它'}
    with open(output, 'w') as out_tsv:
        out_tsv.write('he\ther\tme\tit\n')
        for sentence in zh_sents:
            sent = f'在“{sentence[0]}”这句话中，自己指的是'
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

            softmax_probs = F.softmax(logits, dim=-1)

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

            out_tsv.write(
                f'{next_word_probability_him}\t{next_word_probability_her}\t{next_word_probability_w}\t{next_word_probability_t}\n')
            all_prob = sorted(all_prob.items(), key=lambda x: x[1], reverse=True)

            print(all_prob)
            # if

if __name__ == '__main__':
    test_real_data(natural_local_verb, 'test.tsv')
    test_real_data(natural_long_verb, 'test.tsv')
    test_real_data(natural_long_anim, 'test.tsv')
    # print('In ambiguous setting, the percentage of local binding:')
    # c1, len_sent1 = get_probability(amb_f1, 'result/glm/amb_f1_glm.tsv', female_first=True)
    # c2, len_sent2 = get_probability(amb_m1, 'result/glm/amb_m1_glm.tsv', female_first=False)
    # print('In externally oriented verb setting, the percentage of local binding:')
    # c3, len_sent3 = get_probability(verb_f1, 'result/glm/verb_f1_glm.tsv', female_first=True)
    # c4, len_sent4 = get_probability(verb_m1, 'result/glm/verb_m1_glm.tsv', female_first=False)
    # print('In internally oriented verb setting, the percentage of local binding:')
    # c14, len_sent14 = get_probability(in_verb_f1, 'result/glm/in_verb_f1_glm.tsv', female_first=True)
    # c15, len_sent15 = get_probability(in_verb_m1, 'result/glm/in_verb_m1_glm.tsv', female_first=False)
    # print('In the blocking effect setting, the percentage of local binding:')
    # c5, len_sent5 = get_probability(blocking, 'result/glm/blocking_glm.tsv', blocking=True)
    # print('In animate (noun) setting, the percentage of local binding:')
    # c7, len_sent7 = get_probability(animacy_noun, 'result/glm/animacy_noun_glm.tsv', animacy=True)
    # print('In subject orientation, the percentage of local binding:')
    # c8, len_sent8 = get_probability(subj_f1, 'result/glm/subj_f1_glm.tsv', female_first=False)
    # c9, len_sent9 = get_probability(subj_m1, 'result/glm/subj_m1_glm.tsv', female_first=True)
    # print('In subject orientation in a gender-biased setting, the percentage of local binding:')
    # c10, len_sent10 = get_probability(subj_f1_bias, 'result/glm/subj_f1_bias_glm.tsv', female_first=False)
    # c11, len_sent11 = get_probability(subj_m1_bias, 'result/glm/subj_m1_bias_glm.tsv', female_first=True)
    # print('Local binding percentage:')
    # c12, len_sent12 = get_probability(local_f1, 'result/glm/local_f1.tsv', female_first=False)
    # c13, len_sent13 = get_probability(local_m1, 'result/glm/local_m1.tsv', female_first=True)
    #

