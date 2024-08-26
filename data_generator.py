import random
out_verb=['诬陷','拥抱', '离开', '打扰', '超过', '忘了','回答',
          '表扬','反对','调查','拜访','偷听','暗恋','邀请','等待','嫉妒',
          '羡慕','搭讪','监视','吵醒','激怒','模仿','欺负','追随',
          '回避','思念','仰慕','冷落', '靠近','辅导','干扰','加入','接近','看望', '捞','欢迎','勾引','勾结',
           '等','等候','挤','排挤','欺压','探望','信任','响应','学习',
           '咬','消灭','追赶','追求','阻挡','孝敬','赢','招呼','指导','支援','替换','通知',
           '吸引','影响','杀害','赔偿','衬托','扣留','恐吓','联络','忍受','请示','算计']
# control_verb = ['让']
# speech_verb = ['知道', '反对','记得', '提到', '主张', '埋怨', '表示', '赞成', '希望', '坚信','声称','以为','怀疑','觉得','暗示','表示']
speech_verb  = ['说']
in_verb = ['反思', '反省','坦白', '迷失', '克制', '检讨', '放松', '磨练','奉献','调整', '提升', '炫耀','调节','坚持','交代','计较','检讨','卖弄','表达','表现','扭伤']
amb_verbs = ['喜欢', '厌恶','帮助','相信','支持','欺骗','安慰', '挑战','鄙视','忽略','原谅','怀疑','超越', '设计','讨厌','介绍','夸奖','改造','结束','解放','刺激','考虑','掐','评论','爱惜','爱护','爱','保护', '改正','谈论','维护','牺牲','重复','抓','指着','锻炼','打','操心','纠正','扎','谈到','证明','训练','依靠','优待','鞭策', '尊重','辜负','担心','控制']

inanimate_verbs= ['伤害','改变','影响','吸引','启发','激励','刺激','困扰','鼓舞','鞭策','感动','刺激','迷惑','提醒','激励']
female ='她'
male ='他'
first_person = '我'
second_person = '你'
inanimate_pronoun = '它'
inanimate_nouns = ['这本书','这封信','这个举动','这个结果','这个问题','这句话','这个事件']
inanimate_nouns_no_def = ['书','信','结果','问题','话','成绩','经验', '照片', '计划','意见','想法']
ditransitive_verbs = ['给','送给','告诉','通知','寄给','提供','拒绝回答','提醒','推荐']
ditransitive_verbs_bias = ['推荐','描述','介绍','看','展示']
male_only = ['丈夫', '胳膊', '头发', '身体',  '美貌', '裙子', '胸脯', '面颊', '笑容', '身材', '肉体', '容貌', '处境', '脸庞', '笑声', '举止', '形象', '脸蛋', '愿望', '照片', '关系', '姐姐', '爱人',  '欢心', '前额']
female_only = ['钱',  '情妇', '财产', '印象', '生命', '目光', '敌人', '事业', '伙伴', '球', '媳妇', '床',  '太太', '家庭', '同伙', '妻子', '意见', '女人', '样子', '夫人', '作品', '兄弟', '车', '烟',  '老婆']
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
# def subject_orientation():
#animacy effect
def animacy_effect(noun, verbs, output_file):
    pronouns = ['她','他']
    all_sents = []
    with open(output_file, 'w') as out:
        for n in noun:
            for v in verbs:
                subj = random.choice(pronouns)
                matrix_verb = random.choice(speech_verb)
                output_sent = f'{subj}{matrix_verb}{n}{v}了自己。'
                if output_sent not in all_sents:
                    out.write(f'{output_sent}\n')
                    all_sents.append(output_sent)

def subject_orientation(inanimate_noun_no_def, ditransitive_verb, output_file, female_first=True):
    with open(output_file, 'w') as out:
        for v in ditransitive_verb:
            for n in inanimate_noun_no_def:
                if female_first:
                    sent = '她'+v+'他的'+n+'是关于自己的。'
                    out.write(f'{sent}\n')
                    # sent = '她' + v + '我自己的' + n
                    # out.write(f'{sent}\n')
                else:
                    sent = '他'+ v +'她的'+n+'是关于自己的。'
                    out.write(f'{sent}\n')
                    # sent = '他' + v + '我自己的' + n
                    # out.write(f'{sent}\n')

def subj_gender(inanimate_noun_no_def, ditransitive_verb, output_file, female_first=True):
    with open(output_file, 'w') as out:
        for v in ditransitive_verb:
            for n in inanimate_noun_no_def:
                if female_first:
                    sent = '她'+v+'他的'+n+'是关于自己的。'
                    out.write(f'{sent}\n')
                    # sent = '她' + v + '我自己的' + n
                    # out.write(f'{sent}\n')
                else:
                    sent = '他'+ v +'她的'+n+'是关于自己的。'
                    out.write(f'{sent}\n')
                    # sent = '他' + v + '我自己的' + n
                    # out.write(f'{sent}\n')
def local_binding(amb_verbs, output_file, female=True):
    with open(output_file, 'w') as out:
        for v in amb_verbs:
            if female:
                sent = '她'+v+'自己。'
            else:
                sent = '他' + v + '自己。'
            out.write(f'{sent}\n')


if __name__=='__main__':
    # create_ambiguous_sentences(amb_verbs, output_file='data/amb_f1.txt')
    # create_ambiguous_sentences(amb_verbs, output_file='data/amb_m1.txt', female_first=False)
    # create_verb_pairs(out_verb, output_file='data/verb_f1.txt')
    # create_verb_pairs(out_verb, output_file='data/verb_m1.txt', female_first=False)
    # blocking_effect_generator(amb_verbs, output_file='data/blocking_amb.txt')
    # blocking_effect_generator(out_verb, output_file='data/blocking_verbs.txt')
    create_verb_pairs(in_verb, output_file='data/in_verb_f1.txt')
    create_verb_pairs(in_verb, output_file='data/in_verb_m1.txt', female_first=False)
    # animacy_effect(inanimate_nouns, inanimate_verbs, output_file='data/inanimate_nouns.txt')
    # animacy_effect(inanimate_pronoun, inanimate_verbs, output_file='data/inanimate_pron.txt')
    # subject_orientation(inanimate_nouns_no_def, ditransitive_verbs, output_file='data/subject_orientation_f1.txt')
    # subject_orientation(inanimate_nouns_no_def, ditransitive_verbs, output_file='data/subject_orientation_m1.txt', female_first=False)
    # subject_orientation(female_only, ditransitive_verbs_bias, output_file='data/subject_orientation_f1_bias.txt')
    # subject_orientation(male_only, ditransitive_verbs_bias, output_file='data/subject_orientation_m1_bias.txt', female_first=False)


# local_binding(amb_verbs, output_file='data/local_female.txt', female=True)
    # local_binding(amb_verbs, output_file='data/local_male.txt', female=False)