# zh-reflexive
Project scripts for the course *Computational Discourse Modeling*. 

Please create a virtual environment using ```conda create -n ziji python=3.8``` and install all the dependencies. 
## prepare for the data
- ```data```: contains the designed sentence pairs.
- ```data_generator.py```: the script to generate the sentence pairs

## Probe language models
- ```chatglm.py```: the script used to probe chatglm. 
- ```chinese_bert.py```: the script used to probe chinese-bert-base/multilingual bert/XLM-R.
- ```nmt.py```: the script to probe the Helsinki-NLP zh2en MT system.
