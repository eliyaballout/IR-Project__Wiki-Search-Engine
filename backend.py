from numpy import dot
from numpy.linalg import norm
import numpy as np
from collections import Counter
import pandas as pd
import math
import re
from inverted_index import InvertedIndex

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# --- Bucket name --- #
BUCKET_NAME = "ir_project__bucket1"



def generate_tfidf_vector(query, index: InvertedIndex):
    epsilon = .0000001
    query_vector = np.zeros(len(query))
    counter = Counter(query)
    for token in np.unique(query):
        if token in index.term_total.keys() and token in index.df.keys():
            tf = counter[token] / len(query)
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)
            try:
                ind = query.index(token)
                query_vector[ind] = tf * idf
            except:
                pass
    
    return query_vector


def get_candidate_documents(query, index: InvertedIndex, words):
    candidates = {}
    N = len(index.DL)        
    for term in np.unique(query):  
        if term in words:            
            list_of_doc = index.read_posting_list(term, 'body', BUCKET_NAME)
            normlized_tfidf= []
            
            for doc_id, freq in list_of_doc:
                if doc_id in index.DL.keys():
                    normlized_tfidf.append((doc_id,(freq/index.DL[doc_id])*math.log(N/index.df[term],10)))    
            
            for doc_id, tfidf in normlized_tfidf:
                if tfidf > 0.1:
                    candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf     
    
    return candidates


def generate_document_tfidf_matrix(query, index: InvertedIndex, words):
    candidates_scores = get_candidate_documents(query,index,words)
    unique_candidates = np.unique([doc_id for doc_id, _ in candidates_scores.keys()])
    # now matrix is by length of query
    D = np.zeros((len(unique_candidates), len(query)))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = query

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key  
        D.loc[doc_id][term] = tfidf
    
    return D


def cosine_similarity(doc_matrix, query_vector):
    cos_sims = dict()
    for doc in doc_matrix.axes[0]:
      cos_sim = query_vector.dot( doc_matrix.loc[doc]) / (norm(query_vector) * norm(doc_matrix.loc[doc]))
      cos_sims[doc] = cos_sim
    
    return cos_sims


def get_top_n(sim_dict, N=10):
    docs_and_scores = [(k, v) for k, v in sim_dict.items()]
    return sorted(docs_and_scores, key = lambda x: x[1],reverse=True)[:N]


def get_body_tfidf_score(body_query, body_index: InvertedIndex ,N = 5):
    body_query_tokens = [token.group() for token in RE_WORD.finditer(body_query.lower())]
    words = body_index.term_total.keys()
    Q = generate_tfidf_vector(body_query_tokens, body_index)
    D = generate_document_tfidf_matrix(body_query_tokens, body_index, words)
    cos_sim = cosine_similarity(D,Q)
    topN = get_top_n(cos_sim, N)
    return topN