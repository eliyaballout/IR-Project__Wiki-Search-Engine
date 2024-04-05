from flask import Flask, request, jsonify
from nltk.stem.porter import PorterStemmer
from inverted_index import InvertedIndex
import os
import pickle
import json
from backend import *
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')



#######################################################################################################################
################################################ MyFlaskApp Class #####################################################


class MyFlaskApp(Flask):
  def run(self, host=None, port=None, debug=None, **options):
    super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)



#######################################################################################################################
################################################# Helper Functions ####################################################

# --- Load Index file --- #
def load_index(file_name):
  with open(file_name, 'rb') as file:
    return pickle.load(file)



#######################################################################################################################
################################################# Initializations #####################################################


# --- Bucket name --- #
BUCKET_NAME = "ir_project__bucket1"


# tokenizer and stopwords
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# stemmer
ps = PorterStemmer()


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


# download all indexes
if not os.path.exists('body_index.pkl'):
  body_inv_index = InvertedIndex.read_index('body', 'body_index', BUCKET_NAME)
else:
  body_inv_index = load_index("body_index.pkl")
  print("Body index file already exists.")

if not os.path.exists('title_index.pkl'):
  title_inv_index = InvertedIndex.read_index('title', 'title_index', BUCKET_NAME)
else:
  title_inv_index = load_index("title_index.pkl")
  print("Title index file already exists.")

if not os.path.exists('anchor_index.pkl'):
  anchor_inv_index = InvertedIndex.read_index('anchor', 'anchor_index', BUCKET_NAME)
else:
  anchor_inv_index =  load_index("anchor_index.pkl")
  print("Anchor index file already exists.")


# download page rank to pr varible
if not os.path.exists('pr.json'):
  os.system(f"gsutil cp gs://{BUCKET_NAME}/pr/pr.json .")
else:
  print("Page rank file already exists.")

with open('pr.json') as prJSON:
  pr = json.load(prJSON)


# download titles for each doc 
if not os.path.exists('titles.json'):
  os.system(f"gsutil cp gs://{BUCKET_NAME}/titles/titles.json .")
else:
  print("Titles file already exists.")

with open('titles.json') as titlesJSON:
  titles = json.load(titlesJSON)



#######################################################################################################################
################################################## All Routes #########################################################


@app.route("/search")
def search():
  ''' Returns up to a 100 of your best search results for the query. This is 
    the place to put forward your best search engine, and you are free to
    implement the retrieval whoever you'd like within the bound of the 
    project requirements (efficiency, quality, etc.). That means it is up to
    you to decide on whether to use stemming, remove stopwords, use 
    PageRank, query expansion, etc.

    To issue a query navigate to a URL like:
      http://YOUR_SERVER_DOMAIN/search?query=hello+world
    where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.
  Returns:
  --------
    list of up to 100 search results, ordered from best to worst where each 
    element is a tuple (wiki_id, title).
  '''
  res = []
  query = request.args.get('query', '')
  if len(query) == 0:
    return jsonify(res)
  # BEGIN SOLUTION

  # initialize weights
  body_weight = 0.35
  title_weight = 0.35
  anchor_weight = 0.05
  pr_weight = 0.25

  # get tokens
  query_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
  query_tokens = [token for token in query_tokens if token not in all_stopwords]

  # get body results
  body_results = dict(get_body_tfidf_score(query, body_inv_index, 100))


  # get title results
  docs_and_scores_title = {}

  stems = []
  for token in query_tokens:
    stems.append(ps.stem(token))

  for token in stems:
    docs_and_tfs = title_inv_index.read_posting_list(token, 'title', BUCKET_NAME)
    for (doc,tf) in docs_and_tfs:
        if doc not in docs_and_scores_title.keys():
          docs_and_scores_title[doc] = 1 / len(stems)
        else:
          docs_and_scores_title[doc] = docs_and_scores_title[doc] + (1 / len(stems))
  docs_and_scores_title_list = [(k, v) for k, v in docs_and_scores_title.items()]
  title_results = dict(sorted(docs_and_scores_title_list, key = lambda x: x[1],reverse=True)[:100])

  # get anchor results
  docs_and_scores_anchor = {}
  for token in query_tokens:
    docs_and_tfs = anchor_inv_index.read_posting_list(token, 'anchor', BUCKET_NAME)
    for (doc,tf) in docs_and_tfs:
        if doc not in docs_and_scores_anchor.keys():
          docs_and_scores_anchor[doc] = 1 / len(query_tokens)
        else:
          docs_and_scores_anchor[doc] = docs_and_scores_anchor[doc] + (1 / len(query_tokens))
  docs_and_scores_anchor_list = [(k, v) for k, v in docs_and_scores_anchor.items()]
  anchor_results = dict(sorted(docs_and_scores_anchor_list, key = lambda x: x[1],reverse=True)[:100])

  # all candidates
  all_candidate_docs = set(list(body_results.keys()) + list(title_results.keys()) + list(anchor_results.keys()))

  #searching
  docs_and_scores_final = dict()
  # calculate variables used in searching
  max_body = 1
  if body_results.values():
    max_body = max(body_results.values())

  max_title = 1
  if title_results.values():
    max_title = max(title_results.values())

  max_anchor = 1
  if anchor_results.values():
    max_anchor = max(anchor_results.values())

  prs = []
  for doc in all_candidate_docs:
    if(str(doc) in pr.keys()):
      prs.append(pr[str(doc)])

  max_pr = max(prs)

  # give score for each doc
  for doc in all_candidate_docs:
    body_score = 0
    title_score = 0
    anchor_score = 0
    pr_score = 0

    if(doc in body_results.keys()):
      body_score = body_weight * (body_results[doc]/max_body)

    if(doc in title_results.keys()):
      title_score = title_weight * (title_results[doc]/max_title)

    if(doc in anchor_results.keys()):
      anchor_score = anchor_weight * (anchor_results[doc]/max_anchor)
    
    if(str(doc) in pr.keys()):
      pr_score = pr_weight * (pr[str(doc)]/max_pr)
      
    docs_and_scores_final[doc] = body_score + title_score + anchor_score + pr_score

  res = sorted(list(docs_and_scores_final.items()), key=lambda x: x[1], reverse=True)[:100]
  res = list(map(lambda x: (str(x[0]), titles[str(x[0])]), res))

  # END SOLUTION
  return jsonify(res)
    


########################################################################################
############################## NO NEED TO IMPLEMENT ####################################


@app.route("/search_body")
def search_body():
  ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
    SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
    staff-provided tokenizer from Assignment 3 (GCP part) to do the 
    tokenization and remove stopwords. 

    To issue a query navigate to a URL like:
      http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
    where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.
  Returns:
  --------
    list of up to 100 search results, ordered from best to worst where each 
    element is a tuple (wiki_id, title).
  '''
  res = []
  query = request.args.get('query', '')
  if len(query) == 0:
    return jsonify(res)
  # BEGIN SOLUTION

  # END SOLUTION
  return jsonify(res)


@app.route("/search_title")
def search_title():
  ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
    IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
    DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
    USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
    tokenization and remove stopwords. For example, a document 
    with a title that matches two distinct query words will be ranked before a 
    document with a title that matches only one distinct query word, 
    regardless of the number of times the term appeared in the title (or 
    query). 

    Test this by navigating to the a URL like:
      http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
    where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.
  Returns:
  --------
    list of ALL (not just top 100) search results, ordered from best to 
    worst where each element is a tuple (wiki_id, title).
  '''
  res = []
  query = request.args.get('query', '')
  if len(query) == 0:
    return jsonify(res)
  # BEGIN SOLUTION

  # END SOLUTION
  return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
  ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
    IN THE ANCHOR TEXT of articles, ordered in descending order of the 
    NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
    DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
    3 (GCP part) to do the tokenization and remove stopwords. For example, 
    a document with a anchor text that matches two distinct query words will 
    be ranked before a document with anchor text that matches only one 
    distinct query word, regardless of the number of times the term appeared 
    in the anchor text (or query). 

    Test this by navigating to the a URL like:
      http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
    where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.
  Returns:
  --------
    list of ALL (not just top 100) search results, ordered from best to 
    worst where each element is a tuple (wiki_id, title).
  '''
  res = []
  query = request.args.get('query', '')
  if len(query) == 0:
    return jsonify(res)
  # BEGIN SOLUTION
  
  # END SOLUTION
  return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
  ''' Returns PageRank values for a list of provided wiki article IDs. 

    Test this by issuing a POST request to a URL like:
      http://YOUR_SERVER_DOMAIN/get_pagerank
    with a json payload of the list of article ids. In python do:
      import requests
      requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
    As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.
  Returns:
  --------
    list of floats:
      list of PageRank scores that correrspond to the provided article IDs.
  '''
  res = []
  wiki_ids = request.get_json()
  if len(wiki_ids) == 0:
    return jsonify(res)
  # BEGIN SOLUTION

  # END SOLUTION
  return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
  ''' Returns the number of page views that each of the provide wiki articles
    had in August 2021.

    Test this by issuing a POST request to a URL like:
      http://YOUR_SERVER_DOMAIN/get_pageview
    with a json payload of the list of article ids. In python do:
      import requests
      requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
    As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
    if you're using ngrok on Colab or your external IP on GCP.
  Returns:
  --------
    list of ints:
      list of page view numbers from August 2021 that correrspond to the 
      provided list article IDs.
  '''
  res = []
  wiki_ids = request.get_json()
  if len(wiki_ids) == 0:
    return jsonify(res)
  # BEGIN SOLUTION

  # END SOLUTION
  return jsonify(res)



#######################################################################################################################
################################################## Run The App ########################################################


if __name__ == '__main__':
  # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
  app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)