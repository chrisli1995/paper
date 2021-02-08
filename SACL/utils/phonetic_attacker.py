import pandas as pd
import OpenAttack
import random
import codecs
import json

class PhoneticAttacker(OpenAttack.Attacker):
    def __init__(self, processor = OpenAttack.DefaultTextProcessor()):
        self.processor = processor

    def __call__(self, clsf, x_orig, target=None):
        # Generate a potential adversarial example
        x_new = self.swap(x_orig)
        print(x_new)
        # Get the preidictions of victim classifier
        y_orig, y_new = clsf.get_pred([x_orig, x_new])

        # Check for untargeted or targeted attack
        if (target is None and y_orig != y_new) or target == y_new:
            return x_new, y_new
        else:
            # Failed
            return None

    # test
    def swap(self,sentence):
        f=codecs.open('/home/lwd/cstools/project/TextAattck/my_data/dictionary/words_ps.json','r')
        dictionary=json.load(f)

        tokens = [token for token, pos in self.processor.get_tokens(sentence)]
        index_list=[i for i in range(len(tokens))]
        random.shuffle(index_list)
        for k,i in enumerate(index_list):
            if k==len(index_list)-1:
                tokens[i] = 'unk'
                return self.processor.detokenizer(tokens)
            print('tokens:',tokens[i])
            replace = self.search_word_levenshtein(tokens[i], dictionary)
            print('replace:',replace)
            if len(replace):
                replace_word = replace[0]
                tokens[i] = replace_word
                return self.processor.detokenizer(tokens)
            else:
                continue


    # levenshtein distance
    def ps_matching_levenshtein(self,str1, str2):
        edit = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
        # print(edit)

        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i - 1] == str2[j - 1]:
                    d = 0
                else:
                    d = 1

                edit[i][j] = min(edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + d)

        return int(edit[len(str1)][len(str2)])

    # search word by phonetic symbol
    def search_word_levenshtein(self,target, dictionary, isWordMatching=None):
        try:
            target_ps = dictionary[target]
        except Exception:
            return []
        else:
            print(target_ps)
            min_target_ps_mathcing = 100
            min_target = []
            dictionary.pop(target)
            for k, v in dictionary.items():
                match = self.ps_matching_levenshtein(target_ps, v)
                # print(min_ceshi,match)
                if min_target_ps_mathcing > match:
                    min_target = []
                    min_target.append(k)
                    min_target_ps_mathcing = match
                elif min_target_ps_mathcing == match:
                    min_target.append(k)
            # print(min_target)
            min_target_word = []
            min_target_mathcing = 100
            if isWordMatching:
                for i in min_target:
                    match = self.ps_matching_levenshtein(target, i)
                    if min_target_mathcing > match:
                        min_target_word = []
                        min_target_word.append(i)
                        min_target_mathcing = match
                    elif min_target_mathcing == match:
                        min_target_word.append(i)

            else:
                min_target_word = min_target

        return min_target_word

