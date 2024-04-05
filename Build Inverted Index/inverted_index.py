#imports
from collections import defaultdict, Counter
from contextlib import closing
import pickle
import hashlib
from pathlib import Path
from google.cloud import storage
import itertools
import os
import io



#######################################################################################################################
################################################ Global Variables #####################################################


# --- Block Size --- #
BLOCK_SIZE = 1999998

# --- Reading/Writing params --- #
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1
NUM_BUCKETS = 124



#######################################################################################################################
################################################ Helper Functions #####################################################


# --- hash function --- #
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


# --- token 2 bucket_id --- #
def token2bucket_id(token):
    return int(_hash(token),16) % NUM_BUCKETS



######################################################################################################################
################################################ MultiFileWriter Class ################################################


#writer to GCP
class MultiFileWriter:
    def __init__(self, base_dir, name, bucket_name, ii_name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb') for i in itertools.count())
        self._f = next(self._file_gen)
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.ii_name = ii_name
        
        
    # write to file
    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            if remaining == 0:  
                self._f.close()
                self.upload_to_gcp()                
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        
        return locs


    def close(self):
        self._f.close()
    
    
    # upload file to GCP
    def upload_to_gcp(self):
        self._f.close()
        file_name = self._f.name
        blob = self.bucket.blob(f"{self.ii_name}/{file_name}")
        blob.upload_from_filename(file_name)



#######################################################################################################################
################################################ MultiFileReader Class ################################################

        
#reader from GCP
class MultiFileReader:
    def __init__(self, ii_name, bucket_name):
        self._open_files = {}
        self.ii_name = ii_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)


    # read file from GCP
    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = io.BytesIO(self.bucket.blob(f"{self.ii_name}/{f_name}").download_as_string())
            
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        
        return b''.join(b)
  
  
    def close(self):
        for f in self._open_files.values():
            f.close()


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False 



#######################################################################################################################
############################################### InvertedIndex Class ###################################################


class InvertedIndex:
    
    def __init__(self):
        # stores document length per document
        self.DL = {}
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of (file_name, offset) pairs. 
        self.posting_locs = defaultdict(list)


    # write index variables (not pistings) to GCP
    def write_index(self, base_dir, name):
        self._write_globals(base_dir, name)


    # write index variables
    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)


    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_posting_list']
        return state


    # read index from gcp
    @staticmethod
    def read_index(base_dir, name, bucket_name):
        i_src = f'gs://{bucket_name}/{base_dir}/{name}.pkl'
        gscom = "gsutil cp " + i_src + " ."
        os.system(gscom)
        with open(f'{name}.pkl', 'rb') as f:
            return pickle.load(f)


    # write posting list to GCP
    @staticmethod
    def write_a_posting_list(b_w_pl, bucket_name, ii_name):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl
        with closing(MultiFileWriter('.', bucket_id, bucket_name, ii_name)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big') for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
            
            writer.upload_to_gcp() 
            InvertedIndex._upload_posting_locs(bucket_id, posting_locs, bucket_name, ii_name)
        
        return bucket_id
    
    
    # upload posting locs to GCP
    @staticmethod
    def _upload_posting_locs(bucket_id, posting_locs, bucket_name, ii_name):
        with open(f"{bucket_id}_posting_locs.pickle", "wb") as f:
            pickle.dump(posting_locs, f)
       
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_posting_locs = bucket.blob(f"{ii_name}/{bucket_id}_posting_locs.pickle")
        blob_posting_locs.upload_from_filename(f"{bucket_id}_posting_locs.pickle")


    # read posting list from GCP by given term w
    def read_posting_list(self, w, ii_name, bucket_list):
        if w in self.df.keys() and self.posting_locs.keys():
            with closing(MultiFileReader(ii_name, bucket_list)) as reader:
                locs = self.posting_locs[w]
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                
                return posting_list
        
        return []
    

    # functions to calcualte index variables for RDD
    @staticmethod
    def reduce_word_counts(unsorted_pl):
      return sorted(unsorted_pl, key=lambda x: x[0])


    @staticmethod
    def calculate_df(postings):
      return postings.mapValues(len)


    @staticmethod
    def partition_postings_and_write(postings, bucket_name, ii_name):
      return postings.map(lambda x: (token2bucket_id(x[0]), x)).groupByKey().map(lambda x: InvertedIndex.write_a_posting_list((x[0],x[1]), bucket_name, ii_name))


    @staticmethod
    def get_total_term(tfs):
      tfs = list(tfs)
      return sum(x[1] for x in tfs)