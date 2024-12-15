# zh-reflexive
This repository contains the code for the COLING 2025 Paper *Language Models at the Syntax-Semantics Interface: A Case Study of the Long-Distance Binding of Chinese Reflexive ziji* by Xiulin Yang.


**❗The code will be updated by 31st December when I finish my finals.**

Please create a virtual environment using ```conda create -n ziji python=3.8``` and install all the dependencies. 
## prepare for the data
- ```data```: contains the designed sentence pairs.
- ```data_generator.py```: the script to generate the sentence pairs
- Or you can use the files in the ```data``` folder directly.

## Probe language models
- ```chatglm.py```: the script used to probe chatglm. 
- ```chinese_bert.py```: the script used to probe chinese-bert-base/multilingual bert/XLM-R.
- ```nmt.py```: the script to probe the Helsinki-NLP zh2en MT system.

The results (i.e., the probabilities of all predictions) are stored in the ```probability``` folder. The name of each file is self-explanatory. 
