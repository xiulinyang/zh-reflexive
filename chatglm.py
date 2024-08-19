import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from pathlib import Path
import argparse
import os

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

natural_local_verb = Path('data/real_data_lb_verb.txt').read_text().strip().split('\n')
natural_long_verb = Path('data/real_data_ldb_verb.txt').read_text().strip().split('\n')
natural_long_anim = Path('data/real_data_ldb_anim.txt').read_text().strip().split('\n')
def get_probability(zh_sents, output, blocking = False, female_first=False, animacy=False, verbose=False):
# Get logits from the model
    c=0
    f = []
    m = []
    w = []
    t = []
    with open(output, 'w', encoding="utf-8") as out_tsv:
        out_tsv.write('sent\he\ther\tme\tit\n')
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


            out_tsv.write(f'{sent}\t{next_word_probability_him}\t{next_word_probability_her}\t{next_word_probability_w}\t{next_word_probability_t}\n')
            all_prob = sorted(all_prob.items(), key= lambda x:x[1], reverse=True)
            print(sent)
            print(all_prob)
            if verbose:
                print(sent, all_prob)
            if blocking:
                if all_prob[0][0] =='w':
                    c += 1
            elif animacy:
                if all_prob[0][0] !='t':
                    c+=1
            else:
                if female_first:
                   if all_prob[0][0]=='m':
                        c+=1
                elif not female_first:
                    if all_prob[0][0]=='f':
                        c+=1
        print(f'{c}\t{len(zh_sents)}\t{c / len(zh_sents)}')
    return c, len(zh_sents)


def test_real_data(zh_sents, output, verbose=False):
    c = 0
    f = []
    m = []
    w = []
    t = []
    zh_sents = [x.split() for x in zh_sents]
    label2target = {'f': '她', 'm': '他', 'w': '我', 't': '它'}
    with open(output, 'w',encoding='utf-8') as out_tsv:
        out_tsv.write('sent\tthe\ther\tme\tit\n')
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
                f'{sent}\t{next_word_probability_him}\t{next_word_probability_her}\t{next_word_probability_w}\t{next_word_probability_t}\n')
            all_prob = sorted(all_prob.items(), key=lambda x: x[1], reverse=True)
            print(sent)
            print(all_prob)
            if label2target[all_prob[0][0]] == sentence[1]:
                c+=1
            else:
                if verbose:
                    print(sentence[0])
                    print(all_prob)
                else:
                    pass

        print(f'{c}\t{len(zh_sents)}\t{c / len(zh_sents)}')
    return c, len(zh_sents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='the evaluated model', required=True)
    args = parser.parse_args()
    try:
        os.mkdir(f'result/{args.model}')
    except:
        pass

    natural_long_anim = Path('data/real_data_ldb_anim.txt').read_text().strip().split('\n')
    print('========================REAL DATA==========================================')
    print('real data: local binding, female binder')
    c1, all1 = get_probability(natural_local_f, f'result/{args.model}/natural_local_f1.tsv',  female_first=False)
    print('real data: local binding, male binder')
    c2, all2 = get_probability(natural_local_m, f'result/{args.model}/natural_local_m1.tsv', female_first=True)

    print('real data: reflexive verb, local binding')
    c3, all3 = test_real_data(natural_local_verb, f'result/{args.model}/lb_name.tsv')
    print('real data: non-reflexive verb, long-distance binding')
    c4, all4 = test_real_data(natural_long_verb, f'result/{args.model}/ldb_name.tsv')
    print('real data: animacy effect, long-distance binding')
    c5, all5 = test_real_data(natural_long_anim, f'result/{args.model}/ldb_anim.tsv')

    real_c = c1+c2+c3+c4+c5
    real_all = all1+all2+all3+all4+all5
    print('++++++++++++++++++++++++OVERALL+++++++++++++++++++++++++++++++++++++++++')
    print(f'{real_c}\t{real_all}\t{real_c/real_all}')
    print('========================SYNTHETIC DATA======================================')
    print('In the local binding setting, the percentage of local binding is: ')
    c6, all6 = get_probability(local_f1, f'result/{args.model}/local_f1.tsv',  female_first=False)
    c7, all7 =get_probability(local_m1, f'result/{args.model}/local_m1.tsv',  female_first=True)
    print('In ambiguous setting, the percentage of local binding:')
    c8, all8 =get_probability(amb_f1, f'result/{args.model}/amb_f1.tsv', female_first=True)
    c9, all9 =get_probability(amb_m1, f'result/{args.model}/amb_m1.tsv', female_first=False)
    print('In externally oriented verb setting, the percentage of local binding:')
    c10, all10 =get_probability(verb_f1, f'result/{args.model}/verb_f1.tsv', female_first=True)
    c11, all11 =get_probability(verb_m1, f'result/{args.model}/verb_m1.tsv', female_first=False)
    print('In internally oriented verb setting, the percentage of local binding:')
    c12, all12 =get_probability(in_verb_f1, f'result/{args.model}/in_verb_f1.tsv', female_first=True)
    c13, all13 =get_probability(in_verb_m1, f'result/{args.model}/in_verb_m1.tsv', female_first=False)
    print('In the blocking effect setting, the percentage of local binding:')
    c14, all14 =get_probability(blocking, f'result/{args.model}/blocking.tsv',  blocking=True)
    print('In animate setting, the percentage of long-distance binding:')
    c15, all15 =get_probability(animacy_noun, f'result/{args.model}/animacy_noun.tsv', animacy=True)
    print('In subject orientation, the percentage of local binding:')
    c16, all16 =get_probability(subj_f1, f'result/{args.model}/subj_f1.tsv',  female_first=False)
    c17, all17 =get_probability(subj_m1, f'result/{args.model}/subj_m1.tsv',  female_first=True)
    print('In subject orientation in a gender-biased setting, the percentage of local binding:')
    c18, all18 =get_probability(subj_f1_bias, f'result/{args.model}/subj_f1_bias.tsv',  female_first=False)
    c19, all19 =get_probability(subj_m1_bias, f'result/{args.model}/subj_m1_bias.tsv',  female_first=True)

    print(f'{(c16+c18)/(all16+all18)}\t{(c17+c19)/(all17+all19)}')
    syn_c = c6+c7+c8+c9+all10-c10+all11-c11+c12+c13+c14+c15+c16+c17+c18+c19
    syn_all = all6+all7+all8+all9+all10+all11+all12+all13+all14+all15+all16+all17+all18+all19
    print('+++++++++++++++++++++++OVERALL++++++++++++++++++++++++++')
    print(f'{syn_c}\t{syn_all}\t{syn_c/syn_all}')
