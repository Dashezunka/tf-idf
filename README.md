# Algorithm for creating terms dictionary based on TF-IDF 
It parses articles from different categories to build terms and stop words dictionaries using TF-IDF.

## Description
The algorithm works in the following way:
- Extracts articles of different categories from Medium. 
- Makes corpora for each special category and full corpora with 'ordinary' categories.
- Tokenizes and lemmatizes each article.
- Finds stop words using TF-IDF metrics and builds an appropriate dictionary.
- Builds terms dictionary based on TF-IDF metrics for every special category.
- Saves results in csv-files.

**Note:** For optimal storage of dictionaries a trie structure is used.

## Install and configure
You need to install dependencies from `requirements.txt` using
`pip3 install -r requirements.txt`   

**Note:** If you want to use your own categories, set the CATEGORY_PATH and names of categories in `options.py`.  
You can also adjust stop words and terms thresholds there. 

## Running command
There are two versions of making stop words and terms dictionaries:
- v1 uses only IDF for stop words search and 'threshold' counters for stop words and terms.
- v2 uses whole TF-IDF metrics for stop words and threshold for comparison values of found terms
 with appropriate values in contrast category if this term exists there. 
and sets threshold for average TF-IDF value of contrast category corpus.  
Try `python3 tf_idf.py` or `python3 tf_idf_v2.py` accordingly in the project directory.
