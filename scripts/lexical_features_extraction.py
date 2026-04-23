# lexical_features.py
# -*- coding: utf-8 -*-

import re
from nltk.probability import FreqDist
from pypinyin import pinyin, Style
from collections import Counter

#########################
# 1. 辅助统计函数       #
#########################

def simpsons_diversity_index(tokens):
    fdist = FreqDist(tokens)
    N = sum(fdist.values())
    return 1 - sum((count / N) ** 2 for count in fdist.values())

def calculate_ratio(pattern, text, denominator):
    matched_words = re.findall(pattern, text)
    return len(matched_words) / denominator if denominator != 0 else 0


#############################
# 2. 核心函数：extract_features
#############################

def extract_features(pos_text, stop_words,
                     general_features=True, 
                     ltp_pos_features=True, 
                     additional_features=True, 
                     rhyme_features=True):
    """
    提取"词汇/词性/韵律"特征，返回特征字典。
    """
    features = {}

    # 清理文本，去除词性标注和多余的空格
    tokens = [re.sub(r'(\/\w+\s?)(?!<\/p>)|<[/]?[hp1]{0,2}>', '', token)
              for token in pos_text.split()]
    cleaned_text = ''.join(tokens)

    # ============ 词汇特征 =============
    if general_features:
        num_tokens = len(tokens)
        num_tokens = len(tokens)
        unique_tokens = set(tokens)
        num_unique_tokens = len(unique_tokens)
        fdist = FreqDist(tokens)

        features.update({
            # TTR: 词汇多样性（唯一词数 / 总词数）
            'TTR': num_unique_tokens / num_tokens if num_tokens else 0,

            # MATTR: 滑动窗口 TTR（窗口大小 50）
            'MATTR': (
                sum(len(set(tokens[i:i+50])) / 50 for i in range(0, num_tokens - 50 + 1))
                / (num_tokens - 50 + 1) if num_tokens >= 50 else 0
            ),

            # 平均词长（字符数）
            'Average Word Length': (
                sum(len(token) for token in tokens) / num_tokens if num_tokens else 0
            ),

            # Hapax Legomena Ratio: 词频为 1 的词占比
            'Hapax Legomena Ratio': (
                len([w for w in fdist if fdist[w] == 1]) / num_tokens if num_tokens else 0
            ),

            # 停用词占比
            'Stopwords Ratio': (
                sum(1 for token in tokens if token in stop_words) / num_tokens
                if num_tokens else 0
            ),

            # Simpson's Diversity Index
            'Simpsons Diversity Index': simpsons_diversity_index(tokens),
        })

    # ============ 词性/附加特征部分 =============
    if ltp_pos_features or additional_features:
        total_words = len(re.findall(r'\w+\/', pos_text, re.UNICODE))
        total_punctuations = len(re.findall(r'\/wp', pos_text))
        total_verbs = len(re.findall(r'\w+\/v', pos_text))
        total_conj = len(re.findall(r'\w+\/c', pos_text))
        total_prons = len(re.findall(r'\w+\/r', pos_text))
        total_nouns = len(re.findall(r'\w+\/n', pos_text))
        # 语气助词
        total_yuqi = len(re.findall(r'(的|了|吗|呢|吧|啊|哪|哇|呀|啦|哩)(?=[。！？…])', cleaned_text))

    if ltp_pos_features:
        patterns = {
            # 主要词类
            'ratio_noun':      (r'\w+\/n', total_words, 'pos_text'),
            'ratio_verb':      (r'\w+\/v', total_words, 'pos_text'),
            'ratio_adjective': (r'\w+\/a', total_words, 'pos_text'),
            'ratio_adverb':    (r'\w+\/d', total_words, 'pos_text'),
            'ratio_aux':       (r'\w+\/u', total_words, 'pos_text'),
            'ratio_auxle':     (r'\b了+\b\/u', total_words, 'pos_text'),
            'ratio_prep':      (r'\w+\/p', total_words, 'pos_text'),
            'ratio_baprep': (r'\b把+\b\/p', total_words, 'pos_text'),
            'ratio_beiprep': (r'\b被+\b\/p', total_words, 'pos_text'),
            'ratio_pronoun': (r'\w+\/r', total_words, 'pos_text'),

            # 人称代词（单数）
            'ratio_pronoun':          (r'\w+\/r', total_words, 'pos_text'),
            'ratio_1stPron_singular': (r'\b我\b\/r', total_prons if total_prons != 0 else 1, 'pos_text'),
            'ratio_2ndPron_singular': (r'\b你\b\/r', total_prons if total_prons != 0 else 1, 'pos_text'),
            'ratio_3rdPron_singular': (r'\b(?:他|她|它)\b\/r', total_prons if total_prons != 0 else 1, 'pos_text'),

            # 连词
            'ratio_conn': (r'\w+\/c', total_words, 'pos_text'),
            'ratio_paraConj': (r'(和|与|及|并)\/c', total_conj if total_conj != 0 else 1, 'pos_text'),

            # 连词（续）
            'ratio_advrstvConj': (r'(但|而|但是|然而|可是|不过|否则)\/c', total_conj if total_conj != 0 else 1, 'pos_text'),
            'ratio_causalConj':  (r'(因为|由于|所以|因此|从而)\/c', total_conj if total_conj != 0 else 1, 'pos_text'),

            # 指示代词区分“这”和“那”的单数和复数
            'ratio_quan': (r'\w+\/q', total_words, 'pos_text'),

            'ratio_numeral': (r'\w+\/m', total_words, 'pos_text'),

            'ratio_numeral': (r'\w+\/m', total_words, 'pos_text'),
            'ratio_abbr': (r'\w+\/j', total_words, 'pos_text'),
            'ratio_fornWds': (r'\w+\/ws', total_words, 'pos_text'),
            'ratio_conn': (r'\w+\/c', total_words, 'pos_text'),
            'ratio_paraConj': (r'(和|与|及|并)\/c', total_conj if total_conj != 0 else 1, 'pos_text'),
            'ratio_advrstvConj': (r'(但|而|但是|然而|可是|不过|否则)\/c', total_conj if total_conj != 0 else 1, 'pos_text'),
            'ratio_sequnConj': (r'(然后|于是|随后|才)\/c', total_conj if total_conj != 0 else 1, 'pos_text'),
            'ratio_alterConj': (r'(或者|或|还是|要么|与其|无论)\/c', total_conj if total_conj != 0 else 1, 'pos_text'),
            'ratio_progConj': (r'(不但|不仅|不光|而且|并且|甚至|更|以至|何况|况且|尤其|还|甚至于)\/c', total_conj if total_conj != 0 else 1, 'pos_text'),
            'ratio_quan': (r'\w+\/q', total_words, 'pos_text'),
            'ratio_excl': (r'\w+\/e', total_words, 'pos_text'),
            'ratio_dscrptW': (r'\w+\/z', total_words, 'pos_text'),
            'ratio_idiom': (r'\w+\/i', total_words, 'pos_text'),
            'ratio_period': (r'。+\/wp', total_punctuations if total_punctuations != 0 else 1, 'pos_text'),
            'ratio_qmark': (r'[？?]+\/wp', total_punctuations if total_punctuations != 0 else 1, 'pos_text'),
            'ratio_emark': (r'[！!]+\/wp', total_punctuations if total_punctuations != 0 else 1, 'pos_text'),
            'ratio_comma': (r'[，,]+\/wp', total_punctuations if total_punctuations != 0 else 1, 'pos_text'),
            'ratio_tpadv': (r'(已|将|正|已经|将要|将会|正在)\/', total_words, 'pos_text')
        }

        if additional_features:
            additional_patterns = {
            # 原有的 AA 模式，且排除一些常见称谓或词
                'ratio_AA': (
                    r'(?!太太|妈妈|爸爸|宝宝|奶奶|哥哥|姐姐|斤斤|井井|妹妹|弟弟)([一-龟])\1', 
                    'total_words', 
                    'cleaned_text'
                ),

                'ratio_yuqi': (r'(的|了|吗|呢|吧|啊|哪|哇|呀|啦|哩)(?=[。！？…])', total_words, 'cleaned_text'),
                'ratio_StrongYuqi': (r'(吧|啊|哪|哇|呀|啦)(?=[。！？…])', total_yuqi if total_yuqi != 0 else 1, 'cleaned_text'),
                'ratio_er_suffix': (r'(?<!克|婴|孤|钟|生|女)儿\/', total_words, 'pos_text'),
                'ratio_onomatopoeia': (r'\w+\/o', total_words, 'pos_text'),
                'ratio_metaphor': (r'(?<!(想))(像|如/|好像|仿佛|似乎|类似|犹如|宛如)', total_words, 'pos_text') #去掉如果\想像等词
            }
            patterns.update(additional_patterns)

        ltp_features = {}
        for feature_name, (pattern, denominator, source) in patterns.items():
            text_to_search = pos_text if source == 'pos_text' else cleaned_text
            denominator_value = total_words if denominator == 'total_words' else denominator
            ltp_features[feature_name] = calculate_ratio(pattern, text_to_search, denominator_value)
        features.update(ltp_features)
    return features
