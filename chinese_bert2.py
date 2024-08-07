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
model = 'xlm-roberta-base'


amb_f1 = Path('data/amb_f1.txt').read_text().strip().split('\n')
amb_m1 = Path('data/amb_m1.txt').read_text().strip().split('\n')
verb_f1 = Path('data/verb_f1.txt').read_text().strip().split('\n')
verb_m1 = Path('data/verb_m1.txt').read_text().strip().split('\n')
blocking = Path('data/blocking_amb.txt').read_text().strip().split('\n')
animacy_pro = Path('data/inanimate_pron.txt').read_text().strip().split('\n')
animacy_noun = Path('data/inanimate_nouns.txt').read_text().strip().split('\n')
subj_f1 = Path('data/subject_orientation_f1.txt').read_text().strip().split('\n')
subj_m1 = Path('data/subject_orientation_m1.txt').read_text().strip().split('\n')


local_f1 = Path('data/local_female.txt').read_text().strip().split('\n')
local_m1 = Path('data/local_male.txt').read_text().strip().split('\n')

# Tokenize input
def get_probability(zh_sents, output, local_binding=False, female_first=True, blocking = False, animacy =False):
# Get logits from the model
    nlp = pipeline("fill-mask", model=model)
    mask = nlp.tokenizer.mask_token
    with open(output, 'w') as out_tsv:
        out_tsv.write('he\ther\tme\tit\n')
        c = 0
        target_dic = {'她':'f','他':'m','我':'w','它':'t'}
        target = ['他','她', '我','它']
        for s in tqdm(zh_sents):
            if local_binding:
                text= f'{s[:-3]}{mask}自己。'
            else:
                # de_id = s.index('的')
                # text = f'既然{s}，那么这{s[de_id+1:]}是{mask}的。'
                # text = f'如果{s}, 那么这{s[de_id+1:]}是{mask}的。'
                # print(text)

                text = f'既然{s[:-1]},那么{s[-6:-3]}{mask}自己。'
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


if __name__ == '__main__':
    print('In ambiguous setting, the percentage of local binding:')
    get_probability(amb_f1, 'amb_f1.tsv', female_first=True)
    get_probability(amb_m1, 'amb_m1.tsv', female_first=False)
    print('In externally oriented verb setting, the percentage of local binding:')
    get_probability(verb_f1, 'verb_f1.tsv', female_first=True)
    get_probability(verb_m1, 'verb_m1.tsv',female_first=False)
    print('In the blocking effect setting, the percentage of local binding:')
    get_probability(blocking, 'blocking.tsv', blocking=True)
    print('In animate (pro) setting, the percentage of local binding:')
    get_probability(animacy_pro, 'animacy_pro.tsv', animacy=True)
    print('In animate (noun) setting, the percentage of local binding:')
    get_probability(animacy_noun, 'animacy_noun.tsv', animacy=True)
    # print('In subject orientation, the percentage of local binding:')
    # get_probability(subj_f1, 'subj_f1.tsv', female_first=True)
    # get_probability(subj_m1, 'subj_m1.tsv', female_first=False)
    # print('In the local binding setting, the percentage of local binding is: ')
    # get_probability(local_f1, 'local_f1.tsv', local_binding=True, female_first=False)
    # get_probability(local_m1, 'local_m1.tsv', local_binding=True, female_first=True)
