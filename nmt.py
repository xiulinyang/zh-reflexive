from transformers import MarianTokenizer, MarianMTModel
from pathlib import Path

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

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


ldb = Path('data/real_data_ldb.txt').read_text().strip().split('\n')
lb = Path('data/real_data_lb.txt').read_text().strip().split('\n')

def get_prediction(zh_sents, pred_name, female_first=False, block_first=False, animacy=False):
    srcs = zh_sents
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
            print(src)
            print(detokenised_prds[j])
            print(type(detokenised_prds[j]))
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

    print(c,len(zh_sents))
    print(c/len(zh_sents))
    return c/len(zh_sents)


if __name__ == '__main__':
    print('In ambiguous setting, the percentage of local binding:')
    get_prediction(amb_f1, 'amb_f1.txt', female_first=True)
    get_prediction(amb_m1, 'amb_m1.txt',female_first=False)
    print('In externally oriented verb setting, the percentage of local binding:')
    get_prediction(verb_f1, 'verb_f1.txt', female_first=True)
    get_prediction(verb_m1, 'verb_m1.txt',female_first=False)
    print('In externally blocking effect setting, the percentage of local binding:')
    get_prediction(blocking, 'blocking.txt', block_first=True)
    print('In animate (pro) setting, the percentage of local binding:')
    get_prediction(animacy_pro, 'animacy_pro.txt', animacy=True)
    print('In animate (noun) setting, the percentage of local binding:')
    get_prediction(animacy_noun, 'annimacy_noun.txt', animacy=True)
    print('In subject orientation, the percentage of local binding:')
    get_prediction(subj_f1, 'subj_f1.txt', female_first=True)
    get_prediction(subj_m1,  'subj_m1.txt',female_first=False)
    get_prediction(local_f1, 'local_f1.txt',female_first=False)
    get_prediction(local_m1, 'local_m1.txt',female_first=True)
    # get_prediction(lb, )
