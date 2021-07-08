import numpy as np
import torch

from torch.utils.data import Dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

class CharDict:
    def __init__(self):
        self.word2index = {}  # 字典: 字母->數字
        self.index2word = {}  # 字典: 數字->字典
        self.n_words = 0  # n_words = 28 (26個字母+2個token)

        tokens = ["SOS", "EOS"]
        for t in tokens:
            self.addWord(t)  # token放到字典中

        for i in range(26):
            self.addWord(chr(ord("a") + i))  # 叫出chr中的字母放到字典中

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words  # 建立字典: 字母->數字 {'a': 0}
            self.index2word[self.n_words] = word  # 建立字典: 數字->字典
            self.n_words += 1

    # 建立轉換方式: 單字->數字
    def LongtensorFromString(self, string):
        string = ["SOS"] + list(string) + ["EOS"]
        return torch.LongTensor([self.word2index[ch] for ch in string])

    # 建立轉換方式: 數字->單字
    def StringFromLongtensor(self, longtensor, show_token=False):
        string = ""
        for i in longtensor:

            char = self.index2word[i.item()]

            if len(char) > 1:  # 'SOS' or 'EOS'
                if show_token:
                    char_clean = "<{}>".format(char)
                else:
                    char_clean = ""
            else:
                char_clean = char

            string += char_clean
#             if char == "EOS":
#                 break
        return string


class WordDataset(Dataset):
    def __init__(self, train_=True):

        self.chardict = CharDict()
        self.train = train_

        # load training or test data
        if train_:
            file = "train.txt"
        else:
            file = "test.txt"

        self.data = np.loadtxt(file, dtype=str)  # (1227, 4)

        if train_:
            self.data = self.data.reshape(-1)  # (4908,)
        else:
            # [sp, tp, pg, p]
            self.targets = np.array(
                [
                    [0, 3],  # sp -> p
                    [0, 2],  # sp -> pg
                    [0, 1],  # sp -> tp
                    [0, 1],  # sp -> tp
                    [3, 1],  # p  -> tp
                    [0, 2],  # sp -> pg
                    [3, 0],  # p  -> sp
                    [2, 0],  # pg -> sp
                    [2, 3],  # pg -> p
                    [2, 1],  # pg -> tp
                ]
            )
        self.tenses = ["simple-present", "third-person", "present-progressive", "simple-past"]

    def __len__(self):
        return len(self.data)  # observations, train = 1227, test = 10

    def __getitem__(self, index):
        if self.train:
            c = index % len(self.tenses)  # tense as condition
            v = self.chardict.LongtensorFromString(self.data[index])  # vocabulary
            return v, torch.LongTensor([c])
        else:
            i = self.chardict.LongtensorFromString(self.data[index, 0])
            ci = self.targets[index, 0]
            o = self.chardict.LongtensorFromString(self.data[index, 1])
            co = self.targets[index, 1]

            return i, torch.LongTensor([ci]), o, torch.LongTensor([co])

def gaussian_score(words):
    words_list = []
    score = 0
    yourpath = "train.txt"  # should be your directory of train.txt
    with open(yourpath, "r") as fp:
        for line in fp:
            word = line.split(" ")
            word[3] = word[3].strip("\n")
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score / len(words)

#compute BLEU-4 score
def compute_bleu(reference, output):
    cc = SmoothingFunction()
    return sentence_bleu([reference], output, smoothing_function=cc.method1)


def KL_annealing(iteration, total_iteration): 


#     if iteration > 60000:
#         klw = 0.01/(total_iteration - 60000)*(iteration - 60000)
#         if klw > 0.01:
#             klw = 0.01
#     else:
#         klw = 0

    klw = 0.01

    return klw

def teacher_forcing(iteration, total_iteration):
# 5000 次之後從1遞減至0.5
#     if iteration > 5000:
#         tf_ratio = 1 - (iteration-5000)/(total_iteration-5000)/2
#     else:
#         tf_ratio = 1

    tf_ratio = 0.5
    return tf_ratio