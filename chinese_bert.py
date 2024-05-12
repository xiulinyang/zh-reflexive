from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from scipy.special import softmax
# Load pre-trained model tokenizer (vocabulary) and model
# tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
# model = BertForMaskedLM.from_pretrained('google-bert/bert-base-multilingual-cased')
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

model.eval()  # Put the model in evaluation mode


amb_f1 = Path('data/amb_f1.txt').read_text().strip().split('\n')
amb_m1 = Path('data/amb_m1.txt').read_text().strip().split('\n')
verb_f1 = Path('data/verb_f1.txt').read_text().strip().split('\n')
verb_m1 = Path('data/verb_m1.txt').read_text().strip().split('\n')
blocking = Path('data/blocking_amb.txt').read_text().strip().split('\n')
animacy_pro = Path('data/inanimate_pron.txt').read_text().strip().split('\n')
animacy_noun = Path('data/inanimate_nouns.txt').read_text().strip().split('\n')
subj_f1 = Path('data/subject_orientation_f1.txt').read_text().strip().split('\n')
subj_m1 = Path('data/subject_orientation_m1.txt').read_text().strip().split('\n')
# Tokenize input
def get_probability(zh_sents, output, female_first=True, blocking = False, animacy =False):
# Get logits from the model
    c = 0
    f = []
    m = []
    w = []
    t = []


    with open(output, 'w') as out_tsv:
        out_tsv.write('he\ther\tme\tit\n')
        for s in tqdm(zh_sents):
            text= '在“'+s+'”中，自己指的是<mask>。'
            # print(text)
            tokenized_text = tokenizer.tokenize(text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            mask_index = tokenized_text.index("<mask>")
            # Convert to tensors
            tokens_tensor = torch.tensor([indexed_tokens])

            # Predict all tokens
            with torch.no_grad():
                outputs = model(tokens_tensor)
                predictions = outputs.logits

            # Apply softmax to get probabilities for the masked token
            softmax_probs = F.softmax(predictions[0, mask_index], dim=-1)
            next_word_m = '他'
            next_word_f = '她'
            next_word_w = '我'
            next_word_t = '它'

            next_word_id_m = tokenizer.convert_tokens_to_ids(next_word_m)
            next_word_id_f = tokenizer.convert_tokens_to_ids(next_word_f)
            next_word_id_w = tokenizer.convert_tokens_to_ids(next_word_w)
            next_word_id_t = tokenizer.convert_tokens_to_ids(next_word_t)

            word_probability_m = softmax_probs[next_word_id_m].item()
            word_probability_f = softmax_probs[next_word_id_f].item()
            word_probability_w = softmax_probs[next_word_id_w].item()
            word_probability_t = softmax_probs[next_word_id_t].item()

            # softmax_prob = softmax_probs[word_probability_m, word_probability_f, word_probability_w]
            f.append(word_probability_m)
            m.append(word_probability_f)
            w.append(word_probability_w)
            t.append(word_probability_t)

            all_prob = {'f': word_probability_f, 'm': word_probability_m,
                        'w': word_probability_w, 't': word_probability_t}

            # print(all_prob)
            mm = all_prob['m']
            ff = all_prob['f']
            ww = all_prob['w']
            tt = all_prob['t']
            out_tsv.write(f'{mm}\t{ff}\t{ww}\t{tt}\n')

            all_prob = sorted(all_prob.items(), key=lambda x: x[1], reverse=True)
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
    return c, len(zh_sents)


if __name__ == '__main__':
    print('In ambiguous setting, the percentage of local binding:')
    c1, len_sent1 = get_probability(amb_f1, 'amb_f1_xlm.tsv', female_first=True)
    c2, len_sent2 = get_probability(amb_m1, 'amb_m1_xlm.tsv', female_first=False)
    print('In externally oriented verb setting, the percentage of local binding:')
    c3, len_sent3 = get_probability(verb_f1, 'verb_f1_xlm.tsv', female_first=True)
    c4, len_sent4 = get_probability(verb_m1, 'verb_m1_xlm.tsv', female_first=False)
    print('In the blocking effect setting, the percentage of local binding:')
    c5, len_sent5 = get_probability(blocking, 'blocking_xlm.tsv', blocking=True)
    print('In animate (pro) setting, the percentage of local binding:')
    c6, len_sent6 = get_probability(animacy_pro, 'animacy_pro_xlm.tsv', animacy=True)
    print('In animate (noun) setting, the percentage of local binding:')
    c7, len_sent7 = get_probability(animacy_noun, 'animacy_noun_xlm.tsv', animacy=True)
    print('In subject orientation, the percentage of local binding:')
    c8, len_sent8 = get_probability(subj_f1, 'subj_f1_xlm.tsv', female_first=True)
    c9, len_sent9 = get_probability(subj_m1, 'subj_m1_xlm.tsv', female_first=False)
    c = c1 + c2 + len_sent3 - c3 + len_sent4 - c4 + c5 + len_sent6 - c6 + len_sent7- c7 + len_sent8 - c8 + len_sent9 - c9
    len_sent = len_sent1 + len_sent2 + len_sent3 + len_sent4 + len_sent5 + len_sent6 + len_sent7 + len_sent8 + len_sent9

    print(c, len_sent, c/len_sent)