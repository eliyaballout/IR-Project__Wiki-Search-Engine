# ***Welcome to Wiki Search Engine***



## Introduction

Welcome to Wiki Search Engine, this Information Retrieval project is written in python. <br>
A search engine that allows users to search through Wikipedia articles and find the most relevant results. <br>
This project is a search engine that uses information retrieval techniques to search through Wikipedia articles and find the most relevant results. It utilizes TF-IDF and Cosine Similarity algorithms to rank the results and provide the user with the best match for their query.
<br><br>




## Features

1. **[inverted_index.py](https://github.com/eliyaballout/IR-Project__Wiki-Search-Engine/blob/main/Build%20Inverted%20Index/inverted_index.py):** Inverted index for body, titles, anchors of wiki pages. Also, contains MultiFileWriter and MultiFileReader classes, writing/reading to/from GCP bucket all postings and postings locations from current inverted index. <br><br>

2. **[IndexBuilder.ipynb](https://github.com/eliyaballout/IR-Project__Wiki-Search-Engine/blob/main/Build%20Inverted%20Index/IndexBuilder.ipynb):** Contains all indexing code. Firstly, it gets all wikidata from wikidumps. Then, it creates indexes for body, title (stemmed using Porter Stemmer), anchor. Writes all postings and postings locations (using MultiFileWriter) to GCP, and all inverted indexes data (globals) to GCP. Calculates PageRank and uploads it to GCP to JSON file. Makes id, title JSON file and uploads it to GCP. <br><br>
3. **[search_frontend.py](https://github.com/eliyaballout/IR-Project__Wiki-Search-Engine/blob/main/search_frontend.py):** Main script, flask app, containing all searching logic. <br><br>

4. **[backend.py](https://github.com/eliyaballout/IR-Project__Wiki-Search-Engine/blob/main/backend.py):** Contains all relevant functions for searching:
   1. **body search:** cosine similarity using tf-idf on the body of articles.
   2. **title search:** binary ranking using the title of articles.
   3. **anchor search:** binary ranking using the anchor text. <br><br>

5. **[search_frontend_quality.py](https://github.com/eliyaballout/IR-Project__Wiki-Search-Engine/blob/main/search_frontend_quality.py):** Allowes to test different weights on (title, body, anchor, page rank) and see how different measurements (MAP, Recall, Precision, R-Precision) changes respectively.

<br><br>




## Requirements, Installation & Usage

**I will explain here the requirements, installation and the usage of the search engine:** <br>

**Requirements:**
1. To run this project, you'll need to have Python 3.x installed on your system.

2. Install the required libraries:
   ```
   pip install pyspark
   pip install google-cloud-storage
   ```
<br>


**Installation:**
1. Download and extract the [ZIP file](https://github.com/eliyaballout/IR-Project__Wiki-Search-Engine/archive/refs/heads/main.zip). <br>
   or you can clone the repository to your local machine:
   ```
   git clone https://github.com/eliyaballout/IR-Project__Wiki-Search-Engine.git
   ```

2. Navigate to the project directory.
   ```
   cd IR-Project__Wiki-Search-Engine
   ```

3. Run the search engine.
   ```
   python3 search_frontend.py
   ```

<br>


**Usage:**

After you run the `search_frontend.py`, go to the browser and run:
```
http://127.0.0.1:8080/search?query=YOUR_QUERY
```
where `YOUR_QUERY` should be the query that you want to search for. <br><br>

**Example:** <br>
Query = "take on me"

```
http://127.0.0.1:8080/search?query=take+on+me
```
<br><br>




## Technologies Used
<img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original-wordmark.svg" title="python" alt="python" width="40" height="40"/>&nbsp;
<br>
