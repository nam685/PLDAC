import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyterrier as pt
import re
if not pt.started():
    pt.init()

#marco_indexer = pt.TRECCollectionIndexer('./mini_corpus/MARCO/marco_14000_doc.tsv.index')
#marcoref = marco_indexer.index('./mini_corpus/MARCO/marco_14000_doc.tsv.trecweb')
marcoref = './mini_corpus/MARCO/marco_14000_doc.tsv.index'
marco_index = pt.IndexFactory.of(marcoref)
print(marco_index.getCollectionStatistics().toString())

#kilt_indexer = pt.TRECCollectionIndexer('./mini_corpus/KILT/kilt_9000.index')
#kiltref = kilt_indexer.index('./mini_corpus/KILT/kilt_9000.trecweb')
kiltref = "./mini_corpus/KILT/kilt_9000.index"
kilt_index = pt.IndexFactory.of(kiltref)
print(kilt_index.getCollectionStatistics().toString())

#df_marco_p = pd.read_csv('./mini_corpus/MARCO/marco_300000_passage.tsv',sep='\t',names=['docno','text'],dtype=str)
#marcop_indexer = pt.DFIndexer('./mini_corpus/MARCO/marco_300000_passage.index')
#marcopref = marcop_indexer.index(df_marco_p['text'],df_marco_p['docno'])
marcopref = './mini_corpus/MARCO/marco_300000_passage.index'
marcop_index = pt.IndexFactory.of(marcopref)
print(marcop_index.getCollectionStatistics().toString())

#mk_indexer = pt.TRECCollectionIndexer('./mini_corpus/mk.index')
#mkref = mk_indexer.index(['./mini_corpus/KILT/kilt_9000.trecweb','./mini_corpus/MARCO/marco_14000_doc.tsv.trecweb'])
mkref = './mini_corpus/mk.index'
mk_index = pt.IndexFactory.of(mkref)
print(mk_index.getCollectionStatistics().toString())

marcop_dev_queries = pd.read_csv('./data/marco_passage/queries.dev.tsv', sep='\t', names=['qid', 'query'], dtype=str)
marcop_dev_queries['query'] = marcop_dev_queries['query'].apply(lambda x: re.sub(r'[^A-Za-z0-9]',' ',x))
marcop_dev_qrels = pd.read_csv('./data/marco_passage/qrels.dev.tsv', sep='\t', names=['qid','_','docno', 'label'], dtype= {'qid': str, 'docno': str, 'label': int}).drop(columns=['_'])

bm25_marcop = pt.BatchRetrieve(marcop_index, wmodel="BM25")
#tfidf_marcop = pt.BatchRetrieve(marcop_index, wmodel="TF_IDF")

pt.Experiment(
    [bm25_marcop],
    #[tfidf_marcop,bm25_marcop],
    topics=marcop_dev_queries,
    qrels=marcop_dev_qrels,
    eval_metrics=["map","recip_rank"]
)

