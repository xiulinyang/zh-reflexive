from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
from transformers import logging

logging.set_verbosity_error()
# tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
# model = 'xlm-roberta-base'
# model = 'google-bert/bert-base-chinese'
model = 'google-bert/bert-base-multilingual-cased'
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
# Tokenize input
def get_probability(zh_sents, output, task=None,  female_first=True, blocking = False, animacy =False):
# Get logits from the model
    nlp = pipeline("fill-mask", model=model)
    mask = nlp.tokenizer.mask_token
    with open(output, 'w') as out_tsv:
        out_tsv.write('he\ther\tme\tit\n')
        c = 0
        target_dic = {'她':'f','他':'m','我':'w','它':'t'}
        target = ['他','她', '我','它']
        for s in tqdm(zh_sents):
            if task =='syntax':
                ziji_index = s.index('自')

                text= f'{s[:ziji_index-1]}{mask}{s[ziji_index:]}'
                # print(text)
            elif task =='subject_orientation':
                de_id = s.index('的')
                end_id = s.index('。')
                text = f'如果{s[:-1]}， 那么{s[de_id+1:end_id]}是关于{mask}的。'
            else:
                text = f'如果{s[:-1]},那么{s[2:-3]}{mask}。'
                # text = f'如果{s[:-1]},那么{mask}{s[-5:]}'
            predictions = nlp(text, targets=target)
            # print(predictions)
            all_prob = {target_dic[x['token_str']]: x['score'] for x in predictions}
            m = all_prob['m']
            f = all_prob['f']
            w = all_prob['w']
            t = all_prob['t']
            out_tsv.write(f'{m}\t{f}\t{w}\t{t}\n')


            all_prob = sorted(all_prob.items(), key=lambda x: x[1], reverse=True)
            # print(text, all_prob)
            # print(all_prob)
            if blocking:
                if all_prob[0][0] == 'w':
                    c += 1
            elif animacy:
                if all_prob[0][0] == 't':
                    c += 1
            else:
                if female_first:
                    if all_prob[0][0] == 'm':
                        c += 1
                else:
                    if all_prob[0][0] == 'f':
                        c += 1
        print(c, len(zh_sents))
        print(c/len(zh_sents))

def test_real_data(input_file, output_file):
    input_sents = [x.split() for x in Path(input_file).read_text().strip().split('\n')]
    nlp = pipeline("fill-mask", model=model)
    mask = nlp.tokenizer.mask_token
    with open(output_file, 'w') as out_tsv:
        out_tsv.write('he\ther\tme\tit\n')
        c = 0
        target_dic = {'她': 'f', '他': 'm', '我': 'w', '它': 't'}
        label2target = {'f': '她', 'm': '他', 'w': '我', 't': '它'}
        target = ['他', '她', '我', '它']
        for s in tqdm(input_sents):

            text = f'如果{s[0][:-1]}，那么{s[2][:-2]}{mask}。'

            predictions = nlp(text, targets=target)
            all_prob = {target_dic[x['token_str']]: x['score'] for x in predictions}
            all_prob = sorted(all_prob.items(), key=lambda x: x[1], reverse=True)
            pred = label2target[all_prob[0][0]]

            if pred == s[1]:
                c+=1
            else:
                print(text)
                print(all_prob)


        print(c, len(input_sents))
        print(c/len(input_sents))



if __name__ == '__main__':
    # test_real_data('data/real_data_lb_name.txt', 'test.tsv')
    test_real_data('data/real_data_ldb_verb.txt', 'ldb_name.tsv')
    test_real_data('data/real_data_lb_name.txt', 'lb_inverb.tsv')
    test_real_data('data/real_data_ldb_anim.txt', 'ldb_name.tsv')



    # get_probability(natural_local_f, 'result/cbert/natural_local_f1.tsv', 'syntax', female_first=False)
    # get_probability(natural_local_m, 'result/cbert/natural_local_m1.tsv', 'syntax', female_first=True)

    # print('In the local binding setting, the percentage of local binding is: ')
    # get_probability(local_f1, 'result/cbert/local_f1.tsv', 'syntax', female_first=False)
    # get_probability(local_m1, 'result/cbert/local_m1.tsv', 'syntax', female_first=True)
    # print('In ambiguous setting, the percentage of local binding:')
    # get_probability(amb_f1, 'result/cbert/amb_f1.tsv', female_first=True)
    # get_probability(amb_m1, 'result/cbert/amb_m1.tsv', female_first=False)
    # print('In externally oriented verb setting, the percentage of local binding:')
    # get_probability(verb_f1, 'result/cbert/verb_f1.tsv', female_first=True)
    # get_probability(verb_m1, 'result/cbert/verb_m1.tsv',female_first=False)
    # print('In internally oriented verb setting, the percentage of local binding:')
    # get_probability(in_verb_f1, 'result/cbert/in_verb_f1.tsv', female_first=True)
    # get_probability(in_verb_m1, 'result/cbert/in_verb_m1.tsv', female_first=False)
    # print('In the blocking effect setting, the percentage of local binding:')
    # get_probability(blocking, 'result/cbert/blocking.tsv','syntax', blocking=True)
    # print('In animate setting, the percentage of local binding:')
    # get_probability(animacy_noun, 'result/cbert/animacy_noun.tsv', animacy=True)
    # print('In subject orientation, the percentage of local binding:')
    # get_probability(subj_f1, 'result/cbert/subj_f1.tsv', 'subject_orientation', female_first=False)
    # get_probability(subj_m1, 'result/cbert/subj_m1.tsv', 'subject_orientation', female_first=True)
    # print('In subject orientation in a gender-biased setting, the percentage of local binding:')
    # get_probability(subj_f1_bias, 'result/cbert/subj_f1_bias.tsv', 'subject_orientation', female_first=False)
    # get_probability(subj_m1_bias, 'result/cbert/subj_m1_bias.tsv','subject_orientation', female_first=True)

