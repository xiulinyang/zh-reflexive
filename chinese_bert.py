from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F
from pathlib import Path
# Load pre-trained model tokenizer (vocabulary) and model
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')
model.eval()  # Put the model in evaluation mode


amb_f1 = Path('data/amb_f1.txt').read_text().strip().split('\n')
amb_m1 = Path('data/amb_m1.txt').read_text().strip().split('\n')
verb_f1 = Path('data/verb_f1.txt').read_text().strip().split('\n')
verb_m1 = Path('data/verb_m1.txt').read_text().strip().split('\n')
blocking = Path('data/blocking_amb.txt').read_text().strip().split('\n')
animacy_pro = Path('data/inanimate_pron.txt').read_text().strip().split('\n')
animacy_noun = Path('data/inanimate_nouns.txt').read_text().strip().split('\n')

# Tokenize input
def get_probability(zh_sents, female_first=True, blocking = False, animacy =False):
# Get logits from the model
    c = 0
    f = []
    m = []
    w = []
    t = []
    for s in zh_sents:
        text= '在“'+s+'”中，自己指的是[MASK]。'
        print(text)
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        mask_index = tokenized_text.index("[MASK]")
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
        f.append(word_probability_m)
        m.append(word_probability_f)
        w.append(word_probability_w)
        t.append(word_probability_t)

        all_prob = {'f': word_probability_f, 'm': word_probability_m,
                    'w': word_probability_w, 't': word_probability_t}
        all_prob = sorted(all_prob.items(), key=lambda x: x[1], reverse=True)
        print(all_prob)
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
    print(c/len(zh_sents))


if __name__ == '__main__':
    print('In ambiguous setting, the percentage of long-distance binding:')
    get_probability(amb_f1, female_first=True)
    get_probability(amb_m1, female_first=False)
    print('In externally oriented verb setting, the percentage of long-distance binding:')
    get_probability(verb_f1, female_first=True)
    get_probability(verb_m1, female_first=False)
    print('In the blocking effect setting, the percentage of local binding:')
    get_probability(blocking, blocking=True)
    print('In animate (pro) setting, the percentage of long-distance binding:')
    get_probability(animacy_pro, animacy=True)
    print('In animate (noun) setting, the percentage of long-distance binding:')
    get_probability(animacy_noun, animacy=True)


