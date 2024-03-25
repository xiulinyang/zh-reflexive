import random
out_verb=['诬陷','拥抱','靠近', '离开', '打扰', '超过', '忘了','回答',
          '表扬','反对','调查','拜访','偷听','暗恋','邀请','等待','嫉妒',
          '羡慕','搭讪','欢迎','监视','吵醒','激怒','模仿','欺负','追随',
          '回避','思念','仰慕','冷落','吸引']
control_verb = ['让']
speech_verb = ['知道', '希望', '坚信','声称','以为','怀疑','觉得','预料','推断','暗示','表示']
in_verb = ['反思', '反省','坦白', '提升', '表达','调节','']
amb_verbs = ['喜欢','讨厌', '厌恶','帮助','相信','支持','欺骗','安慰',
            '挑战','鄙视','忽略','原谅','怀疑','超越','表扬',
            '鞭策', '尊重','辜负','担心','控制']
inanimate_nouns =['树','窗户','展出',]
female ='她'
male ='他'
first_person = '我'
second_person = '你'


# ambiguous
def create_ambiguous_sentences(amb_verbs, output_file,female_first=True):
    with open(output_file, 'w') as out:
        for amb_verb in amb_verbs:
            matrix_verb = random.choice(speech_verb)
            if female_first:
                out_sent = female+matrix_verb+male+amb_verb+'自己。'
            else:
                out_sent = male + matrix_verb + female + amb_verb + '自己。'

            out.write(f'{out_sent}\n')

# verb effect
def create_verb_pairs(out_verbs, output_file,female_first=True):
    with open(output_file, 'w') as out:
        for amb_verb in out_verbs:
            matrix_verb = random.choice(speech_verb)
            if female_first:
                out_sent = female+matrix_verb+male+amb_verb+'自己。'
            else:
                out_sent = male + matrix_verb + female + amb_verb + '自己。'

            out.write(f'{out_sent}\n')

# blocking effect
def blocking_effect_generator(verbs, output_file):
    with open(output_file, 'w') as out:
        for amb_verb in verbs:
            matrix_verb = random.choice(speech_verb)
            for subj in [female, male]:
                out_sent = subj+matrix_verb+'我'+amb_verb+'自己。'
            out.write(f'{out_sent}\n')


# subject/speaker orientation

# animacy effect

if __name__=='__main__':
    create_ambiguous_sentences(amb_verbs, output_file='data/amb_f1.txt')
    create_ambiguous_sentences(amb_verbs, output_file='data/amb_m1.txt', female_first=False)
    create_verb_pairs(out_verb, output_file='data/verb_f1.txt')
    create_verb_pairs(out_verb, output_file='data/verb_m1.txt', female_first=False)
    blocking_effect_generator(amb_verbs, output_file='data/blocking_amb.txt')
    blocking_effect_generator(out_verb, output_file='data/blocking_verbs.txt')