from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from pathlib import Path
import argparse
import os
from scipy.special import softmax
from collections import Counter



local_f1 = Path('data/local_female.txt').read_text().strip().split('\n')
local_m1 = Path('data/local_male.txt').read_text().strip().split('\n')
amb_f1 = Path('data/amb_f1.txt').read_text().strip().split('\n')
amb_m1 = Path('data/amb_m1.txt').read_text().strip().split('\n')
verb_f1 = Path('data/verb_f1.txt').read_text().strip().split('\n')
verb_m1 = Path('data/verb_m1.txt').read_text().strip().split('\n')
in_verb_f1 = Path('data/in_verb_f1.txt').read_text().strip().split('\n')
in_verb_m1 = Path('data/in_verb_m1.txt').read_text().strip().split('\n')
blocking = Path('data/blocking_literature.txt').read_text().strip().split('\n')
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
natural_blocking = Path('data/real_data_blocking.txt').read_text().strip().split('\n')
def get_probability(zh_sents, output, task=None, llm=False, antecedent = None, antecedent_list = None, verbose=False):
# Get logits from the model
    c=0
    target_dic = {'她': 'f', '他': 'm', '我': 'w', '它': 't', '你': 'n'}
    with open(output, 'w', encoding="utf-8") as out_tsv:
        for text in zh_sents:
            if '他自己' in text or '她自己' in text:
                ziji_index = text.index('自')
                text = f'{text[:ziji_index]}{text[ziji_index:]}'

            if task == 'subject_orientation':
                de_id = text.index('的')
                # end_id = text.index('是')
                end_id = len(text)-1
                sent = f'如果{text[:-1]}， 那么{text[de_id + 1:end_id]}是关于'
            elif task == 'verb_orientation':
                sent = f'如果{text[:-1]},那么{text[2:-3]}'
            else:
                sent = f'如果{text[:-1]},那么{text[:-3]}'

            if llm:
                sent = f'在“{text}”中，自己指的是'

            encoded_input = tokenizer(sent, return_tensors='pt').to(model.device)

            with torch.no_grad():
                outputs = model(**encoded_input)
                logits = outputs.logits  # Assuming the model outputs include logits

            target_dic = {x:y for x, y in target_dic.items() if y in antecedent_list}
            next_word_ids = {x:tokenizer.encode(x, add_special_tokens=False)[0] for x, y in target_dic.items()}
            softmax_probs = F.softmax(logits, dim=-1)
            all_prob = {y: softmax_probs[0, -1, next_word_ids[x]].item() for x, y in target_dic.items()}
            scores = softmax([y for _,y in all_prob.items()])
            preds = [x for x, _ in all_prob.items()]
            all_prob = {x: y for x, y in zip(preds, scores)}
            prob_to_write = '\t'.join([x + ':' + str(y) for x, y in all_prob.items()])
            out_tsv.write(f'{sent}\t{prob_to_write}\n')
            all_prob = sorted(all_prob.items(), key=lambda x: x[1], reverse=True)

            if verbose:
                print(sent, all_prob)

            if antecedent == 'w':
                if all_prob[0][0] == 'w':
                    c += 1
            elif antecedent == 'f':
                if all_prob[0][0] == 'f':
                    c += 1
            elif antecedent == 'm':
                if all_prob[0][0] == 'm':
                    c += 1
            elif antecedent == 'n':
                if all_prob[0][0] == 'n':
                    c += 1
            else:
                if all_prob[0][0] != 't':
                    c += 1
        print(f'{c}\t{len(zh_sents)}\t{c / len(zh_sents)}')
    return c, len(zh_sents)


def test_real_data(input_file, output, task = None, verbose=False):
    c = 0
    zh_sents = [x.split() for x in Path(input_file).read_text().strip().split('\n')]
    target2label = {'她':'f', '他':'m', '我':'w', '它':'t'}
    with open(output, 'w',encoding='utf-8') as out_tsv:
        for sentence in zh_sents:
        
            sent = f'如果{sentence[0][:-1]}，那么{sentence[2][:-2]}'

            encoded_input = tokenizer(sent, return_tensors='pt').to(model.device)
            with torch.no_grad():
                outputs = model(**encoded_input)
                logits = outputs.logits
            freq_char = Counter(sent)
            if task =='animacy':
                to_add_antecedent = ['她','他','它']
            elif task == 'blocking':
                to_add_antecedent = ['她','他','我','你']
            else:
                to_add_antecedent = ['她', '他']
            antecedent_list = list(set([x for x, y in target2label.items() if freq_char[x] > 0 or x in to_add_antecedent]))
            
            target_dic = {x: y for x, y in target2label.items() if x in antecedent_list}
            
            label2target = {y:x for x, y in target_dic.items()}
            next_word_ids = {x: tokenizer.encode(x, add_special_tokens=False)[0] for x, y in target_dic.items()}
            softmax_probs = F.softmax(logits, dim=-1)
            all_prob = {y: softmax_probs[0, -1, next_word_ids[x]].item() for x, y in target_dic.items()}
            scores = softmax([y for _, y in all_prob.items()])
            preds = [x for x, _ in all_prob.items()]
            all_prob = {x: y for x, y in zip(preds, scores)}
            prob_to_write = '\t'.join([x + ':' + str(y) for x, y in all_prob.items()])
            out_tsv.write(f'{sent}\t{prob_to_write}\n')
            all_prob = sorted(all_prob.items(), key=lambda x: x[1], reverse=True)

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
    llm = False

    if args.model =='gpt-distil':
        tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
        model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")

    elif args.model =='gpt':
        tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
        model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

    elif args.model =='gpt-large':
        tokenizer = BertTokenizer.from_pretrained("uer/gpt2-large-chinese-cluecorpussmall")
        model = GPT2LMHeadModel.from_pretrained("uer/gpt2-large-chinese-cluecorpussmall")
    elif args.model == 'gpt-medium':
        tokenizer = BertTokenizer.from_pretrained("uer/gpt2-medium-chinese-cluecorpussmall")
        model = GPT2LMHeadModel.from_pretrained("uer/gpt2-medium-chinese-cluecorpussmall")
    elif args.model =='gpt-xxl':
        tokenizer = BertTokenizer.from_pretrained("uer/gpt2-xlarge-chinese-cluecorpussmall")
        model = GPT2LMHeadModel.from_pretrained("uer/gpt2-xlarge-chinese-cluecorpussmall")
    elif args.model =='glm9b':
        tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True).half().cuda()
        llm = True
    else:
        print(args.model)
        raise ValueError('The model name is not recognized')

    model.eval()



    # print('========================REAL DATA==========================================')
    # print('real data: local binding, female binder')
    # c1, all1 = get_probability(natural_local_f, f'result/{args.model}/natural_local_f1.tsv',  antecedent='f',
    #                            antecedent_list=['f', 'm'])
    # print('real data: local binding, male binder')
    # c2, all2 = get_probability(natural_local_m, f'result/{args.model}/natural_local_m1.tsv', antecedent='m',
    #                            antecedent_list=['f', 'm'])
    #
    # print('real data: reflexive verb, local binding')
    # c3, all3 = test_real_data('data/real_data_lb_verb.txt', f'result/{args.model}/lb_name.tsv')
    # print('real data: non-reflexive verb, long-distance binding')
    # c4, all4 = test_real_data('data/real_data_ldb_verb.txt', f'result/{args.model}/ldb_name.tsv')
    # print('real data: animacy effect, long-distance binding')
    # c5, all5 = test_real_data('data/real_data_ldb_anim.txt', f'result/{args.model}/ldb_anim.tsv', task='animacy')
    #
    # print('real data: blocking effect, long-distance binding')
    # c20, all20 = test_real_data('data/real_data_blocking.txt', f'result/{args.model}/natural_blocking.tsv',
    #                             task='blocking')
    #
    # real_c = c1 + c2 + c3 + c4 + c5 + c20
    # real_all = all1 + all2 + all3 + all4 + all5 + all20
    # print('++++++++++++++++++++++++OVERALL+++++++++++++++++++++++++++++++++++++++++')
    # print(f'{real_c}\t{real_all}\t{real_c / real_all}')
    print('========================SYNTHETIC DATA======================================')
    print('In the local binding setting, the percentage of local binding is: ')
    c6, all6 = get_probability(local_f1, f'result/{args.model}/local_f1.tsv',  antecedent='f', llm=llm,
                               antecedent_list=['f', 'm'])
    c7, all7 = get_probability(local_m1, f'result/{args.model}/local_m1.tsv', antecedent='m', llm=llm,
                               antecedent_list=['f', 'm'])
    print('In ambiguous setting, the percentage of local binding:')
    c8, all8 = get_probability(amb_f1, f'result/{args.model}/amb_f1.tsv', task='verb_orientation', llm=llm,antecedent='f', antecedent_list=['f', 'm'])
    c9, all9 = get_probability(amb_m1, f'result/{args.model}/amb_m1.tsv', task='verb_orientation', llm=llm, antecedent='m', antecedent_list=['f', 'm'])
    print((c8 + c9) / (all8 + all9))
    print('In externally oriented verb setting, the percentage of local binding:')
    c10, all10 = get_probability(verb_f1, f'result/{args.model}/verb_f1.tsv', task='verb_orientation', llm=llm, antecedent='f',
                                 antecedent_list=['f', 'm'])
    c11, all11 = get_probability(verb_m1, f'result/{args.model}/verb_m1.tsv', task='verb_orientation', llm=llm,antecedent='m',
                                 antecedent_list=['f', 'm'])
    print((c10 + c11) / (all10 + all11))
    print('In internally oriented verb setting, the percentage of local binding:')
    c12, all12 = get_probability(in_verb_f1, f'result/{args.model}/in_verb_f1.tsv', task='verb_orientation', llm=llm,antecedent='f',
                                 antecedent_list=['f', 'm'])
    c13, all13 = get_probability(in_verb_m1, f'result/{args.model}/in_verb_m1.tsv', task='verb_orientation',llm=llm, antecedent='m',
                                 antecedent_list=['f', 'm'])
    print((c12 + c13) / (all12 + all13))
    print('In the blocking effect setting, the percentage of local binding:')
    c14, all14 = get_probability(blocking, f'result/{args.model}/blocking.tsv', task='verb_orientation',llm=llm,antecedent='w',
                                 antecedent_list=['f', 'm', 'w'])
    print('In animate setting, the percentage of long-distance binding:')
    c15, all15 = get_probability(animacy_noun, f'result/{args.model}/animacy_noun.tsv', task='verb_orientation', llm=llm, antecedent='t',
                                 antecedent_list=['f', 'm', 't'])
    print('In subject orientation, the percentage of local binding:')
    c16, all16 = get_probability(subj_f1, f'result/{args.model}/subj_f1.tsv', 'subject_orientation', llm=llm,antecedent='f',
                                 antecedent_list=['f', 'm'])
    c17, all17 = get_probability(subj_m1, f'result/{args.model}/subj_m1.tsv', 'subject_orientation', llm=llm,antecedent='m',
                                 antecedent_list=['f', 'm'])
    # print('In subject orientation in a gender-biased setting, the percentage of local binding:')
    # c18, all18 =get_probability(subj_f1_bias, f'result/{args.model}/subj_f1_bias.tsv', 'subject_orientation', antecedent='f', antecedent_list=['f','m'])
    # c19, all19 =get_probability(subj_m1_bias, f'result/{args.model}/subj_m1_bias.tsv', 'subject_orientation', antecedent='m', antecedent_list=['f','m'])

    # print(f'{(c16+c18)/(all16+all18)}\t{(c17+c19)/(all17+all19)}')
    syn_c = c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15 + c16 + c17
    syn_all = all6 + all7 + all8 + all9 + all10 + all11 + all12 + all13 + all14 + all15 + all16 + all17
    print('+++++++++++++++++++++++OVERALL++++++++++++++++++++++++++')
    print(f'{syn_c}\t{syn_all}\t{syn_c / syn_all}')

    # print(
    #     f'{round((c1 / all1) * 100, 1)}&{round((c2 / all2) * 100, 1)}&{round((c4 / all4) * 100, 1)}&{round((c3 / all3) * 100, 1)}&{round((c20 / all20), 1)}&{round((c5 / all5) * 100, 1)}&{round((real_c / real_all) * 100, 1)}')
    print(
        f'{round((c6 / all6) * 100, 1)}&{round((c7 / all7) * 100, 1)}&{round(((c10 + c11) / (all10 + all11)) * 100, 1)}&{round(((c12 + c13) / (all12 + all13)) * 100, 1)}&{round((c14 / all14) * 100, 1)}&{round(((c8 + c9) / (all8 + all9)) * 100, 1)}&{round((c15 / all15) * 100, 1)}&{round((c16 / all16) * 100, 1)}&{round((c17 / all17) * 100, 1)}&{round((syn_c / syn_all) * 100, 1)}')
