# Copied from notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def forget_history(l,k,max_length=1000):
    title = l[0]
    section_title = l[1]
    k = int(min(k, (len(l)-2)/2))
    forget_index = int(len(l) - k*2)
    past_queries = [l[i] for i in range(2,forget_index,2)][::-1]
    near_queries = [l[i] for i in range(forget_index,len(l),2)][::-1]
    near_answers = [l[i] for i in range(forget_index+1,len(l),2)][::-1]
    context = []
    # Estimate number of tokens = 3 * number of characters
    length = len(title) + len(section_title)
    for i in range(len(near_queries)):
        length += len(near_queries[i]) + len(near_queries[i])
        if length >= max_length:
            break
        context.append(near_queries[i])
        context.append(near_answers[i])
    for i in range(len(past_queries)):
        length += len(past_queries[i])
        if length >= max_length:
            break
        context.append(past_queries[i])
    return [title,section_title] + context

def to_csv(df,dest):
    df.rename(columns={"Rewrite":"Target"},inplace=True)
    df = pd.DataFrame(df[["Question","History","Target"]])
    df["Source"] = "Query: " + df["Question"] + " |||| Context: " + df["History"] + " |||| Reformulation: "
    df.drop(columns=["Question","History"], inplace=True)
    df.to_csv(dest)

df_canard_dev = pd.read_json('./data/canard/dev.json').drop(columns=["QuAC_dialog_id","Question_no"])
df_canard_dev_all_3 = df_canard_dev.copy()
df_canard_dev_all_2 = df_canard_dev.copy()
df_canard_dev_all_1 = df_canard_dev.copy()
df_canard_dev_all_0 = df_canard_dev.copy()

df_canard_dev["History"] = df_canard_dev["History"].apply(lambda x: " ||| ".join(forget_history(x,k=np.inf)))
df_canard_dev_all_3["History"] = df_canard_dev_all_3["History"].apply(lambda x: " ||| ".join(forget_history(x,k=3)))
df_canard_dev_all_2["History"] = df_canard_dev_all_2["History"].apply(lambda x: " ||| ".join(forget_history(x,k=2)))
df_canard_dev_all_1["History"] = df_canard_dev_all_1["History"].apply(lambda x: " ||| ".join(forget_history(x,k=1)))
df_canard_dev_all_0["History"] = df_canard_dev_all_0["History"].apply(lambda x: " ||| ".join(forget_history(x,k=0)))


df_canard_train = pd.read_json('./data/canard/train.json').drop(columns=["QuAC_dialog_id","Question_no"])
df_canard_train_all_3 = df_canard_train.copy()
df_canard_train_all_2 = df_canard_train.copy()
df_canard_train_all_1 = df_canard_train.copy()
df_canard_train_all_0 = df_canard_train.copy()

df_canard_train["History"] = df_canard_train["History"].apply(lambda x: " ||| ".join(forget_history(x,k=np.inf)))
df_canard_train_all_3["History"] = df_canard_train_all_3["History"].apply(lambda x: " ||| ".join(forget_history(x,k=3)))
df_canard_train_all_2["History"] = df_canard_train_all_2["History"].apply(lambda x: " ||| ".join(forget_history(x,k=2)))
df_canard_train_all_1["History"] = df_canard_train_all_1["History"].apply(lambda x: " ||| ".join(forget_history(x,k=1)))
df_canard_train_all_0["History"] = df_canard_train_all_0["History"].apply(lambda x: " ||| ".join(forget_history(x,k=0)))


df_canard_test = pd.read_json('./data/canard/test.json').drop(columns=["QuAC_dialog_id","Question_no"])
df_canard_test_all_3 = df_canard_test.copy()
df_canard_test_all_2 = df_canard_test.copy()
df_canard_test_all_1 = df_canard_test.copy()
df_canard_test_all_0 = df_canard_test.copy()

df_canard_test["History"] = df_canard_test["History"].apply(lambda x: " ||| ".join(forget_history(x,k=np.inf)))
df_canard_test_all_3["History"] = df_canard_test_all_3["History"].apply(lambda x: " ||| ".join(forget_history(x,k=3)))
df_canard_test_all_2["History"] = df_canard_test_all_2["History"].apply(lambda x: " ||| ".join(forget_history(x,k=2)))
df_canard_test_all_1["History"] = df_canard_test_all_1["History"].apply(lambda x: " ||| ".join(forget_history(x,k=1)))
df_canard_test_all_0["History"] = df_canard_test_all_0["History"].apply(lambda x: " ||| ".join(forget_history(x,k=0)))


to_csv(df_canard_dev,"./data/canard/dev_all_all.csv")
to_csv(df_canard_dev_all_3,"./data/canard/dev_all_3.csv")
to_csv(df_canard_dev_all_2,"./data/canard/dev_all_2.csv")
to_csv(df_canard_dev_all_1,"./data/canard/dev_all_1.csv")
to_csv(df_canard_dev_all_0,"./data/canard/dev_all_0.csv")

to_csv(df_canard_train,"./data/canard/train_all_all.csv")
to_csv(df_canard_train_all_3,"./data/canard/train_all_3.csv")
to_csv(df_canard_train_all_2,"./data/canard/train_all_2.csv")
to_csv(df_canard_train_all_1,"./data/canard/train_all_1.csv")
to_csv(df_canard_train_all_0,"./data/canard/train_all_0.csv")

to_csv(df_canard_test,"./data/canard/test_all_all.csv")
to_csv(df_canard_test_all_3,"./data/canard/test_all_3.csv")
to_csv(df_canard_test_all_2,"./data/canard/test_all_2.csv")
to_csv(df_canard_test_all_1,"./data/canard/test_all_1.csv")
to_csv(df_canard_test_all_0,"./data/canard/test_all_0.csv")

df_canard_test = pd.read_csv("./data/canard/test_all_all.csv")
df_canard_test_all_3 = pd.read_csv("./data/canard/test_all_3.csv")
df_canard_test_all_2 = pd.read_csv("./data/canard/test_all_2.csv")
df_canard_test_all_1 = pd.read_csv("./data/canard/test_all_1.csv")
df_canard_test_all_0 = pd.read_csv("./data/canard/test_all_0.csv")

df_canard_train = pd.read_csv("./data/canard/train_all_all.csv")
df_canard_train_all_3 = pd.read_csv("./data/canard/train_all_3.csv")
df_canard_train_all_2 = pd.read_csv("./data/canard/train_all_2.csv")
df_canard_train_all_1 = pd.read_csv("./data/canard/train_all_1.csv")
df_canard_train_all_0 = pd.read_csv("./data/canard/train_all_0.csv")

df_canard_dev = pd.read_csv("./data/canard/dev_all_all.csv")
df_canard_dev_all_3 = pd.read_csv("./data/canard/dev_all_3.csv")
df_canard_dev_all_2 = pd.read_csv("./data/canard/dev_all_2.csv")
df_canard_dev_all_1 = pd.read_csv("./data/canard/dev_all_1.csv")
df_canard_dev_all_0 = pd.read_csv("./data/canard/dev_all_0.csv")

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#!wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
#!unzip ./wiki-news-300d-1M.vec.zip

import io
from collections import defaultdict

def default_vector():
    return np.zeros(300)

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = defaultdict(default_vector)
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:]).astype(float)
    return data

w2v = load_vectors("/content/gdrive/MyDrive/PLDAC/wiki-news-300d-1M.vec")

#!pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')

def sim_bert(u,ps,model=model,threshold=0.25):
    sim = util.dot_score(model.encode(u),model.encode(ps)).numpy()[0]
    return sim - threshold

import spacy

def custom_sentencizer(doc):
    ''' Look for sentence start tokens by scanning for periods. '''
    for i, token in enumerate(doc[:-2]):
        if token.text == ".":
            doc[i+1].is_sent_start = True
#        else:
#            doc[i+1].is_sent_start = False
    return doc

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(custom_sentencizer, before="parser")

def sentence_segmentation(passage):
    return [s.text for s in nlp(passage).sents]

stemmer = SnowballStemmer("english")
key_tags = set(["CD","FW","JJ","JJR","JJS","NN","NNS","NNP","NNPS","VB","VBD","VBG","VBN","VBP","VBZ"])

def extract(s,stem=False,lower=True):
    # (to lower), tokenize, then filter by tag and remove stopword, (then stem)
    if lower:
        s = s.lower()
    if stem:
        return [stemmer.stem(w) for (w,tag) in nltk.pos_tag(nltk.word_tokenize(s)) if tag in key_tags and w not in stopwords.words('english')]
    else:
        return [w for (w,tag) in nltk.pos_tag(nltk.word_tokenize(s)) if tag in key_tags and w not in stopwords.words('english')]

def sim_overlap(u,ss):
    # number of keywords in common
    return [len(set(extract(s,True,False)) & set(extract(u,True,False))) for s in ss]

def sim_words(vu,vs,threshold,w2v=w2v):
    return np.dot(w2v[vu],w2v[vs]) - threshold if vu in w2v and vs in w2v else 0

def sim_maxsim(u,ss,threshold=1.2):
    # my idea
    sim = np.array([ np.mean([\
                        np.max([\
                            sim_words(vu,vs,threshold)\
                        for vs in set(extract(s))]+[0])\
                   for vu in set(extract(u))]+[0])\
                for s in ss])
    return sim


def key_sentence(p,u_next,u,sim):
    # sentence with highest similarity. Note: argmax favors early sentences in case of hesitation, which is a desirable behaviour.
    
    ss = sentence_segmentation(p)
    if len(ss) == 0:
        return ""
    scores = sim(u_next,ss)
    if np.max(scores,initial=-1) > 0:
        return ss[np.argmax(scores)]
    else:
        scores = sim(u,ss)
        if np.max(scores,initial=-1) > 0:
            return ss[np.argmax(scores)]
    return ""

class TRECSet():
    def __init__(self, data, year):
        self.topics = [Topic(topic) for topic in data]
        #self.write_history_simple()
        #self.write_history_overlap_extraction()
        #self.write_history_word2vec_extraction()
        #self.write_history_bert_extraction()
        self.year = year
    
    def to_csv(self,start=1,stop=17):
        self.df = pd.concat([topic.to_df() for topic in self.topics])
        for i in range(start,stop):
            df_i = pd.DataFrame(self.df[["Question",f"History{i}","Target"]])
            df_i["Source"] = df_i.apply(lambda x: "Query: " + x["Question"] + " |||| Context: " + str(x[f"History{i}"]) + " |||| Reformulation: ", axis=1)
            df_i.drop(columns=["Question",f"History{i}"], inplace=True)
            df_i.to_csv(f"./data/treccast/treccastweb-master/{self.year}/trec{self.year}_{i}.csv")
        self.df.to_csv(f"./data/treccast/treccastweb-master/{self.year}/trec{self.year}.csv")
    
    def write_history_simple(self):
        for topic in self.topics:
            topic.write_history_simple()
        self.to_csv(1,5)

    def write_history_overlap_extraction(self):
        for topic in self.topics:
            topic.write_history_overlap_extraction()
        self.to_csv(5,9)

    def write_history_word2vec_extraction(self):
        for topic in self.topics:
            topic.write_history_word2vec_extraction()
        self.to_csv(9,13)

    def write_history_bert_extraction(self):
        for topic in self.topics:
            topic.write_history_bert_extraction()
        self.to_csv(13,17)

class Topic():
    def __init__(self, topic):
        self.number = topic["number"]
        self.turns = [Turn(turn) for turn in topic["turn"]]
        self.MAX_HISTORY_LENGTH = 1000
    
    def to_df(self):
        return pd.DataFrame([turn.to_array() for turn in self.turns],\
                columns=["Question","Target","History1","History2","History3",\
                "History4","History5","History6","History7","History8",\
                "History9","History10","History11","History12","History13",\
                "History14","History15","History16"])

    def write_history_simple(self):
        history1 = []
        history2 = []
        history3 = []
        history4 = []
        for i in range(len(self.turns)):
            # 1> query only
            self.turns[i].history1 = " ||| ".join(history1[::-1])[:self.MAX_HISTORY_LENGTH]
            history1.append(self.turns[i].raw_utterance)
            # 2> recursive query only
            self.turns[i].history2 = " ||| ".join(history2[::-1])[:self.MAX_HISTORY_LENGTH]
            history2.append(self.turns[i].manual_rewritten_utterance)
            # 3> naive
            self.turns[i].history3 = " ||| ".join(history3[::-1])[:self.MAX_HISTORY_LENGTH]
            history3.append(self.turns[i].passage)
            history3.append(self.turns[i].raw_utterance)
            # 4> recursive, top 2 summary
            self.turns[i].history4 = " ||| ".join(history4[::-1])[:self.MAX_HISTORY_LENGTH]
            history4.append(" ".join(sentence_segmentation(self.turns[i].passage)[:2]))
            history4.append(self.turns[i].manual_rewritten_utterance)

    def write_history_overlap_extraction(self):
        history5 = []
        history6 = []
        history7 = []
        history8 = []

        for i in range(len(self.turns)):
            # key sentence overlap extraction
            overlap_extracted_sentence = key_sentence(self.turns[i].passage,\
                self.turns[i+1].raw_utterance if i+1<len(self.turns) else "",\
                self.turns[i].raw_utterance, sim=sim_overlap)

            # 5> recursive, overlap key sentence extraction, last 1 paragraphs
            self.turns[i].history5 = " ||| ".join(history5[::-1])[:self.MAX_HISTORY_LENGTH]
            if (i-1)*2 >= 0:
                history5[(i-1)*2] = ""
            history5.append(overlap_extracted_sentence)
            history5.append(self.turns[i].manual_rewritten_utterance)
            
            # 6> recursive, overlap key sentence extraction, last 2 paragraphs
            self.turns[i].history6 = " ||| ".join(history6[::-1])[:self.MAX_HISTORY_LENGTH]
            if (i-2)*2 > 0:
                history6[(i-2)*2] = ""
            history6.append(overlap_extracted_sentence)
            history6.append(self.turns[i].manual_rewritten_utterance)

            # 7> recursive, overlap key sentence extraction, last 3 paragraphs
            self.turns[i].history7 = " ||| ".join(history7[::-1])[:self.MAX_HISTORY_LENGTH]
            if (i-3)*2 > 0:
                history7[(i-3)*2] = ""
            history7.append(overlap_extracted_sentence)
            history7.append(self.turns[i].manual_rewritten_utterance)

            # 8> recursive, overlap key sentence extraction, all paragraphs
            self.turns[i].history8 = " ||| ".join(history8[::-1])[:self.MAX_HISTORY_LENGTH]
            history8.append(overlap_extracted_sentence)
            history8.append(self.turns[i].manual_rewritten_utterance)

    def write_history_word2vec_extraction(self):
        history9 = []
        history10 = []
        history11 = []
        history12 = []
        for i in range(len(self.turns)):
            # key sentence word2vec maxsim extraction
            maxsim_extracted_sentence = key_sentence(self.turns[i].passage,\
                self.turns[i+1].raw_utterance if i+1<len(self.turns) else "",\
                self.turns[i].manual_rewritten_utterance, sim=sim_maxsim)
            
            # 9> maxsim key sentence extraction, last paragraph
            self.turns[i].history9 = " ||| ".join(history9[::-1])[:self.MAX_HISTORY_LENGTH]
            if (i-1)*2 >= 0:
                history9[(i-1)*2] = ""
            history9.append(maxsim_extracted_sentence)
            history9.append(self.turns[i].manual_rewritten_utterance)
            
            # 10> maxsim key sentence extraction, 2 last paragraph
            self.turns[i].history10 = " ||| ".join(history10[::-1])[:self.MAX_HISTORY_LENGTH]
            if (i-2)*2 >= 0:
                history10[(i-2)*2] = ""
            history10.append(maxsim_extracted_sentence)
            history10.append(self.turns[i].manual_rewritten_utterance)

            # 11> maxsim key sentence extraction, 3 last paragraph
            self.turns[i].history11 = " ||| ".join(history11[::-1])[:self.MAX_HISTORY_LENGTH]
            if (i-3)*2 >= 0:
                history11[(i-3)*2] = ""
            history11.append(maxsim_extracted_sentence)
            history11.append(self.turns[i].manual_rewritten_utterance)

            # 12> maxsim key sentence extraction, all paragraph
            self.turns[i].history12 = " ||| ".join(history12[::-1])[:self.MAX_HISTORY_LENGTH]
            history12.append(maxsim_extracted_sentence)
            history12.append(self.turns[i].manual_rewritten_utterance)

    def write_history_bert_extraction(self):
        history13 = []
        history14 = []
        history15 = []
        history16 = []
        for i in range(len(self.turns)):
            # key sentence bert sim extraction
            bert_extracted_sentence = key_sentence(self.turns[i].passage,\
                self.turns[i+1].raw_utterance if i+1<len(self.turns) else "",\
                self.turns[i].manual_rewritten_utterance, sim=sim_bert)
            
            # 13> bert key sentence extraction, last paragraph
            self.turns[i].history13 = " ||| ".join(history13[::-1])[:self.MAX_HISTORY_LENGTH]
            if (i-1)*2 >= 0:
                history13[(i-1)*2] = ""
            history13.append(bert_extracted_sentence)
            history13.append(self.turns[i].manual_rewritten_utterance)
            

            # 14> bert key sentence extraction, 2 last paragraph
            self.turns[i].history14 = " ||| ".join(history14[::-1])[:self.MAX_HISTORY_LENGTH]
            if (i-2)*2 >= 0:
                history14[(i-2)*2] = ""
            history14.append(bert_extracted_sentence)
            history14.append(self.turns[i].manual_rewritten_utterance)

            # 15> bert key sentence extraction, 3 last paragraph
            self.turns[i].history15 = " ||| ".join(history15[::-1])[:self.MAX_HISTORY_LENGTH]
            if (i-3)*2 >= 0:
                history15[(i-3)*2] = ""
            history15.append(bert_extracted_sentence)
            history15.append(self.turns[i].manual_rewritten_utterance)

            # 16> bert key sentence extraction, all paragraph
            self.turns[i].history16 = " ||| ".join(history16[::-1])[:self.MAX_HISTORY_LENGTH]
            history16.append(bert_extracted_sentence)
            history16.append(self.turns[i].manual_rewritten_utterance)

class Turn():
    def __init__(self, turn):
        self.number = turn["number"]
        self.raw_utterance = turn["raw_utterance"]
        self.passage = turn["passage"]
        self.manual_rewritten_utterance = turn["manual_rewritten_utterance"]
        self.manual_query_history = []
        self.raw_query_history = []
        self.passage_history = []
        self.history1 = ""
        self.history2 = ""
        self.history3 = ""
        self.history4 = ""
        self.history5 = ""
        self.history6 = ""
        self.history7 = ""
        self.history8 = ""
        self.history9 = ""
        self.history10 = ""
        self.history11 = ""
        self.history12 = ""
        self.history13 = ""
        self.history14 = ""
        self.history15 = ""
        self.history16 = ""

    def to_array(self):
        return [self.raw_utterance, self.manual_rewritten_utterance,\
                self.history1, self.history2, self.history3, self.history4,\
                self.history5, self.history6, self.history7, self.history8,\
                self.history9, self.history10, self.history11, self.history12,\
                self.history13, self.history14, self.history15, self.history16]

import json

with open("./data/treccast/treccastweb-master/2021/2021_manual_evaluation_topics_v1.0.json") as json_file:
    data_2021 = json.load(json_file)

treccast_2021 = TRECSet(data_2021,2021)
treccast_2021.write_history_simple()
treccast_2021.write_history_overlap_extraction()
treccast_2021.write_history_word2vec_extraction()
treccast_2021.write_history_bert_extraction()
treccast_2021.to_csv()

with open("/content/gdrive/MyDrive/PLDAC/data/treccast/treccastweb-master/2020/2020_manual_evaluation_topics_v1.0.json") as json_file:
    data_2020 = json.load(json_file)

passages_2020 = dict()
with open("/content/gdrive/MyDrive/PLDAC/mini_corpus/MARCO/marco_passage_2020.tsv","r") as infile:
    lines = infile.readlines()
    for line in lines:
        tmp = line.strip().split(None, 1)
        if len(tmp) > 0:
            id, text = tmp
            passages_2020[id] = text

for topic in data_2020:
    for turn in topic['turn']:
        passage_id = turn['manual_canonical_result_id']
        if passage_id[:6] == "MARCO_":
            turn['passage'] = passages_2020[passage_id[6:]]
        else:
            # I don't have the CAR dataset
            turn['passage'] = ""

treccast_2020 = TRECSet(data_2020,2020)
treccast_2020.write_history_simple()
treccast_2020.write_history_overlap_extraction()
treccast_2020.write_history_word2vec_extraction()
treccast_2020.write_history_bert_extraction()
treccast_2020.to_csv()

