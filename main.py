# !pip install numpy panda torch langchain-text-splitters sentence-transformers ipykernel rank_bm25 faiss-cpu ranx ragatouille==0.0.8
# Import libraries
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from ragatouille import RAGPretrainedModel
from rank_bm25 import BM25Okapi
from ranx import Qrels, Run, fuse, evaluate

def text_preprocess(text):
    text = text.lower()
    return text

def build_run(results_df, doc_id_col="chunk_id"):
    run_df = results_df.copy()
    run = Run.from_df(
        df=run_df,
        q_id_col="query_id",
        doc_id_col=doc_id_col,
        score_col="score",
    )
    return run

# Prepare data
## Load data
texts_df = pd.read_csv("texts.csv")
queries_df = pd.read_csv("queries.csv")

## Chunking configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=64,
    length_function=len,
    is_separator_regex=False,
)

## Split data and process chunks
doc_ids = []
chunk_ids = []
chunk_texts = []
for _, row in texts_df.iterrows():
    doc_id = str(row["doc_id"])
    doc_chunk_texts = text_splitter.split_text(row["doc_text"])
    n_chunk_texts = len(doc_chunk_texts)
    doc_chunk_ids = [f"{doc_id}-{str(i)}" for i in range(n_chunk_texts)]
    # basic text processing
    doc_chunk_texts = [text_preprocess(chunk) for chunk in doc_chunk_texts]
    # save results
    doc_ids.extend([doc_id] * n_chunk_texts)
    chunk_ids.extend(doc_chunk_ids)
    chunk_texts.extend(doc_chunk_texts)

## Save results as df
df = pd.DataFrame({
    "doc_id": doc_ids,
    "chunk_id": chunk_ids,
    "chunk_text": chunk_texts,
})


# Indexing data
## BM25
tokenized_corpus = [doc.split(" ") for doc in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)

## Sentence Transformers + Faiss Index
sent_model = SentenceTransformer("all-mpnet-base-v2")
sent_embeddings = sent_model.encode(chunk_texts)
nb, d = sent_embeddings.shape
index = faiss.IndexFlatL2(d)
index.add(sent_embeddings)

## RAGatuille + Colbert
try:
    RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/index")
except:
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    RAG.index(
        index_name="index", 
        collection=chunk_texts, 
        document_ids=chunk_ids, 
        use_faiss=False,
        max_document_length=1024,
        split_documents=False,
    )


# Search Functions
## BM 25
def bm25_search(query_text, bm25=bm25, df=df):
    query_text = text_preprocess(query_text)
    tokenized_query = query_text.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    bm25_df = df.copy()
    bm25_df["score"] = doc_scores
    bm25_df = bm25_df.loc[bm25_df["score"] > 0]
    # Drop to get docs, no chunks
    bm25_df = bm25_df.sort_values("score", ascending=False)
    bm25_df = bm25_df.drop_duplicates(subset=["doc_id"], keep="first")
    # Return results
    return bm25_df

# Sentence vector search
def sentsim_search(query_text, index=index, df=df, k=5):
    sent_query_emb = sent_model.encode(query_text).reshape(1,-1)
    D, I = index.search(sent_query_emb, k)
    sent_df = df.loc[I[0]]
    sent_df["score"] = D[0].astype(float)
    # Drop to get docs, no chunks
    sent_df = sent_df.sort_values("score", ascending=False)
    sent_df = sent_df.drop_duplicates(subset=["doc_id"], keep="first")
    return sent_df

# Colbert search
def colbert_search(query_text, RAG=RAG, df=df):
    # Run query
    colbert_results = RAG.search(query_text)
    # Save results as a df
    colbert_df = pd.DataFrame(colbert_results)
    colbert_df = colbert_df.merge(df, how="left", left_on="document_id", right_on="chunk_id")
    # Drop to get docs, no chunks
    colbert_df = colbert_df.sort_values("score", ascending=False)
    colbert_df = colbert_df.drop_duplicates(subset=["doc_id"], keep="first")
    return colbert_df

# Fusion rank search
def combined_search(query_text, norm="min-max", method="max", df=df):
    bm25_df = bm25_search(query_text)
    sent_df = sentsim_search(query_text)
    colbert_df = colbert_search(query_text)
    runs = []
    # Save results in Run format
    for run_df in [bm25_df, sent_df, colbert_df]:
        # Adds a random query id since it's necessary for the run
        run_df["query_id"] = "0"
        run_df["chunk_id"] = run_df["chunk_id"].astype(str)
        run = build_run(run_df)
        runs.append(run)
    ## Combining scores
    combined_run = fuse(
        runs=runs,
        norm=norm,
        method=method,
    )
    combined_run_df = combined_run.to_dataframe()
    combined_run_df = combined_run_df.drop("q_id", axis=1)
    combined_run_df = combined_run_df.rename({"doc_id": "chunk_id"}, axis=1)
    combined_run_df = combined_run_df.merge(df, how="left", on="chunk_id")
    # Drop to get docs, no chunks
    combined_run_df = combined_run_df.sort_values("score", ascending=False)
    combined_run_df = combined_run_df.drop_duplicates(subset=["doc_id"], keep="first")
    ## Return similar format to other responses
    return combined_run_df


def search(query_text, mode="bm25"):
    if mode=="bm25":
        results_df = bm25_search(query_text)
    elif mode=="sentsim":
        results_df = sentsim_search(query_text)
    elif mode=="colbert":
        results_df = colbert_search(query_text)
    elif mode=="combined":
        results_df = combined_search(query_text)
    return results_df


def evaluate_search(mode="bm25", queries_df=queries_df):
    # Create Qrel for evaluation
    queries_df["query_id"] = queries_df["query_id"].astype(str)
    queries_df["doc_id"] = queries_df["doc_id"].astype(str)
    queries_df.loc[queries_df["score"] > 0, "score"] = 1
    # Replace all positive score with 1 for simplicity
    qrels = Qrels.from_df(
        df=queries_df,
        q_id_col="query_id",
        doc_id_col="doc_id",
        score_col="score",
    )

    results_list = []
    queries_list = queries_df[["query_id", "query_text"]].drop_duplicates().values.tolist()
    # Get search responses
    for query_id, query_text in queries_list:
        response_df = search(query_text, mode=mode)
        response_df["query_id"] = query_id
        results_list.append(response_df)

    results_df = pd.concat(results_list)
    results_df["doc_id"] = results_df["doc_id"].astype(str)
    run_results = build_run(results_df, doc_id_col="doc_id")

    ## Evaluate runs
    metrics = evaluate(qrels, run_results, ["ndcg@5", "mrr"])
    print(mode, metrics)

evaluate_search(mode="bm25")
evaluate_search(mode="sentsim")
evaluate_search(mode="colbert")
evaluate_search(mode="combined")
