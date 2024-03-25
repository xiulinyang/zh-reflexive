from transformers import MarianTokenizer, MarianMTModel
from pathlib import Path

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

amb_f1 = Path('data/amb_f1.txt').read_text().strip().split('\n')
amb_m1 = Path('data/amb_m1.txt').read_text().strip().split('\n')
verb_f1 = Path('data/verb_f1.txt').read_text().strip().split('\n')
verb_m1 = Path('data/verb_m1.txt').read_text().strip().split('\n')
def get_prediction(zh_sents, female_first=True):
    srcs = zh_sents
    batch = tokenizer.prepare_seq2seq_batch(src_texts=srcs, return_tensors="pt")
    if female_first:
        outputs_beam = model.generate(
            **batch, num_beams=5, max_length=512,
            return_dict_in_generate=True, output_hidden_states=True,
            output_attentions=True)

        detokenised_prds = tokenizer.batch_decode(
            outputs_beam["sequences"], skip_special_tokens=True)

        for j, src in enumerate(srcs):
            print(src)
            print(detokenised_prds[j])


if __name__ == '__main__':
    get_prediction(amb_m1, female_first=True)
    get_prediction(verb_m1, female_first=True)
    get_prediction(verb_f1, female_first=True)
