# Install required libraries 
# !pip install numpy panda torch langchain-text-splitters sentence-transformers ipykernel rank_bm25 faiss-cpu ranx ragatouille==0.0.8

# Import libraries
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from ragatouille import RAGPretrainedModel
from rank_bm25 import BM25Okapi
from ranx import Qrels, Run, fuse, evaluate

# Utility functions
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
texts_df = pd.read_csv("data/texts.csv")
queries_df = pd.read_csv("data/queries.csv")

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
chunks_df = pd.DataFrame({
    "doc_id": doc_ids,
    "chunk_id": chunk_ids,
    "chunk_text": chunk_texts,
})


# Indexing data
## BM25
bm_25_tokenized_corpus = [chunk.split(" ") for chunk in chunk_texts]
bm25_index = BM25Okapi(bm_25_tokenized_corpus)

## Sentence Transformers + Faiss Index
sentsim_model = SentenceTransformer("all-mpnet-base-v2")
sentsim_embeddings = sentsim_model.encode(chunk_texts)
sentsim_embedding_size = sentsim_embeddings.shape[1]
sentsim_index = faiss.IndexFlatL2(sentsim_embedding_size)
sentsim_index.add(sentsim_embeddings)

## RAGatuille + Colbert
if __name__ == "__main__":   # Required so ragatouille runs safely
    try:
        colbert_index = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/index")
    except:
        colbert_index = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        colbert_index.index(
            index_name="index", 
            collection=chunk_texts, 
            document_ids=chunk_ids, 
            use_faiss=False,
            max_document_length=1024,
            split_documents=False,
        )


# Search Functions
## BM 25
def bm25_search(query_text, bm25_index=bm25_index, chunks_df=chunks_df):
    # Preprocess query same as docs
    query_text = text_preprocess(query_text)
    # Transform query
    tokenized_query = query_text.split(" ")
    # Search with bm25 index
    doc_scores = bm25_index.get_scores(tokenized_query)
    # Format as dataframe
    bm25_df = chunks_df.copy()
    bm25_df["score"] = doc_scores
    bm25_df = bm25_df.loc[bm25_df["score"] > 0]
    # Drop to get docs, no chunks
    bm25_df = bm25_df.sort_values("score", ascending=False)
    bm25_df = bm25_df.drop_duplicates(subset=["doc_id"], keep="first")
    # Return results
    return bm25_df

# Sentence vector search
def sentsim_search(query_text, sentsim_model=sentsim_model, sentsim_index=sentsim_index, chunks_df=chunks_df, k=5):
    # Preprocess query same as docs
    query_text = text_preprocess(query_text)
    # Encode query
    sentsim_query_emb = sentsim_model.encode(query_text).reshape(1,-1)
    # Search with embedding
    D, I = sentsim_index.search(sentsim_query_emb, k)
    # Format as dataframe
    sentsim_df = chunks_df.copy()
    sentsim_df = sentsim_df.loc[I[0]]
    sentsim_df["score"] = D[0].astype(float)
    # Drop to get docs, no chunks
    sentsim_df = sentsim_df.sort_values("score", ascending=False)
    sentsim_df = sentsim_df.drop_duplicates(subset=["doc_id"], keep="first")
    return sentsim_df

# Colbert search
def colbert_search(query_text, colbert_index=colbert_index, chunks_df=chunks_df, k=5):
    # Preprocess query same as docs
    query_text = text_preprocess(query_text)
    # Run query
    colbert_results = colbert_index.search(query_text, k=k)
    # Save results as a df
    colbert_df = pd.DataFrame(colbert_results)
    colbert_df = colbert_df.rename({"document_id": "chunk_id"}, axis=1)
    colbert_df = colbert_df.merge(chunks_df, how="left", on="chunk_id")
    colbert_df = colbert_df[["doc_id", "chunk_id", "chunk_text", "score"]]
    # Drop to get docs, no chunks
    colbert_df = colbert_df.sort_values("score", ascending=False)
    colbert_df = colbert_df.drop_duplicates(subset=["doc_id"], keep="first")
    return colbert_df

# Fusion rank search
def combined_search(query_text, fusion_norm="min-max", fusion_method="max", chunks_df=chunks_df):
    runs = []
    for search_fun in [bm25_search, sentsim_search, colbert_search]:
        # Save results in Run format
        run_df = search_fun(query_text)
        run_df["query_id"] = "0"   # query id is required for the run
        run = build_run(run_df)
        runs.append(run)
    ## Combining runs
    combined_run = fuse(
        runs=runs,
        norm=fusion_norm,
        method=fusion_method,
    )
    ## Saving as dataframe
    combined_df = combined_run.to_dataframe()
    combined_df = combined_df.drop("q_id", axis=1)
    combined_df = combined_df.rename({"doc_id": "chunk_id"}, axis=1)
    combined_df = combined_df.merge(chunks_df, how="left", on="chunk_id")
    combined_df = combined_df[["doc_id", "chunk_id", "chunk_text", "score"]]
    # Drop to get docs, no chunks
    combined_df = combined_df.sort_values("score", ascending=False)
    combined_df = combined_df.drop_duplicates(subset=["doc_id"], keep="first")
    ## Return similar format to other responses
    return combined_df

# Defines a global search function 
def search(query_text, mode="bm25"):
    if mode=="bm25":
        return bm25_search(query_text)
    elif mode=="sentsim":
        return sentsim_search(query_text)
    elif mode=="colbert":
        return colbert_search(query_text)
    elif mode=="combined":
        return combined_search(query_text)

# Evaluates search results based on labeled queries 
def evaluate_search(mode="bm25", queries_df=queries_df):
    # Preprocess df
    queries_df["query_id"] = queries_df["query_id"].astype(str)
    queries_df["doc_id"] = queries_df["doc_id"].astype(str)
    queries_df.loc[queries_df["score"] > 0, "score"] = 1   # Replace all positive scores with 1
    # Create Qrel for evaluation
    qrels = Qrels.from_df(
        df=queries_df,
        q_id_col="query_id",
        doc_id_col="doc_id",
        score_col="score",
    )

    # Get search responses
    unique_queries_df = queries_df[["query_id", "query_text"]].drop_duplicates()
    unique_queries_list = unique_queries_df.values.tolist()
    responses_list = []
    for query_id, query_text in unique_queries_list:
        response_df = search(query_text, mode=mode)
        response_df["query_id"] = query_id
        responses_list.append(response_df)

    # Build run dataframe
    run_df = pd.concat(responses_list)
    run_df["doc_id"] = run_df["doc_id"].astype(str)
    run = build_run(run_df, doc_id_col="doc_id")

    ## Evaluate run
    metrics = evaluate(qrels, run, ["f1", "mrr"])
    print(mode, metrics)

# Evaluate bm25
evaluate_search(mode="bm25")
# Evaluate single vector sentence similarity
evaluate_search(mode="sentsim")
# Evaluate colbert
evaluate_search(mode="colbert")
# Evaluate fused search
evaluate_search(mode="combined")
