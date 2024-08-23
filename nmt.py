from transformers import MarianTokenizer, MarianMTModel
from pathlib import Path
import argparse
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

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

natural_block = Path('data/real_data_blocking.txt').read_text().strip().split('\n')
ldb = Path('data/real_data_ldb.txt').read_text().strip().split('\n')
lb = Path('data/real_data_lb.txt').read_text().strip().split('\n')



def get_prediction(zh_sents, pred_name, model = None,female_first=False, block_first=False, animacy=False, verbose=False):
    srcs = [x.split()[0] for x in zh_sents]
    c =0
    batch = tokenizer.prepare_seq2seq_batch(src_texts=srcs, return_tensors="pt")

    outputs_beam = model.generate(
            **batch, num_beams=5, max_length=512,
            return_dict_in_generate=True, output_hidden_states=True,
            output_attentions=True)

    detokenised_prds = tokenizer.batch_decode(
            outputs_beam["sequences"], skip_special_tokens=True)

    with open(pred_name, 'w') as trans:
        for j, src in enumerate(srcs):
            print(src, detokenised_prds[j])
            trans.write(f'{src}\t{detokenised_prds[j]}\n')
            if block_first:
                if 'my' in detokenised_prds[j]:
                    c += 1
            elif animacy:
                if 'it' in detokenised_prds[j] or 'it' in detokenised_prds[j].split()[-1]:
                    c += 1
            else:
                if female_first:
                    if 'himself' in detokenised_prds[j] or 'his' in detokenised_prds[j]:
                        c += 1

                else:
                    if 'herself' in detokenised_prds[j] or 'his' not in detokenised_prds[j]:
                        c += 1
            if verbose:
                print(f'{src}\t{detokenised_prds[j]}')

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
    if args.model == 'nmt':
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    elif args.model == 'mbart':
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    print('In local binding setting, the percentage of local binding:')
    get_prediction(natural_local_f, 'result/nmt/natural_local_f1.txt',female_first=False)
    get_prediction(natural_local_m, 'result/nmt/natural_local_m1.txt',female_first=True)

    get_prediction(natural_local_verb, 'result/nmt/natural_local_verb.txt', female_first=True)
    get_prediction(natural_long_verb, 'result/nmt/natural_long_verb.txt', female_first=True)
    get_prediction(natural_long_anim, 'result/nmt/natural_long_anim.txt', female_first=True)
    get_prediction(natural_block, 'result/nmt/natural_blocking.txt', female_first=False)
    print('In local binding setting, the percentage of local binding:')
    get_prediction(local_f1, 'result/nmt/local_f1.txt',female_first=False)
    get_prediction(local_m1, 'result/nmt/local_m1.txt',female_first=True)
    print('In ambiguous setting, the percentage of local binding:')
    get_prediction(amb_f1, 'result/nmt/amb_f1.txt', female_first=True)
    get_prediction(amb_m1, 'result/nmt/amb_m1.txt',female_first=False)
    print('In externally oriented verb setting, the percentage of local binding:')
    get_prediction(verb_f1, 'result/nmt/verb_f1.txt', female_first=True)
    get_prediction(verb_m1, 'result/nmt/verb_m1.txt',female_first=False)
    print('In internally oriendted verb setting, the percentage of local binding:')
    get_prediction(in_verb_f1, 'result/nmt/in_verb_f1.txt', female_first=True)
    get_prediction(in_verb_m1, 'result/nmt/in_verb_m1.txt',female_first=False)

    print('In blocking effect setting, the percentage of local binding:')
    get_prediction(blocking, 'result/nmt/blocking.txt', block_first=True)
    print('In animate (noun) setting, the percentage of local binding:')
    get_prediction(animacy_noun, 'result/nmt/annimacy_noun.txt', animacy=True)
    print('In subject orientation, the percentage of local binding:')
    get_prediction(subj_f1, 'result/nmt/subj_f1.txt', female_first=False)
    get_prediction(subj_m1,  'result/nmt/subj_m1.txt',female_first=True)

    print('In subject orientation, the percentage of local binding:')
    get_prediction(subj_f1_bias, 'result/nmt/subj_f1_bias.txt', female_first=False)
    get_prediction(subj_m1_bias, 'result/nmt/subj_m1_bias.txt', female_first=True)

