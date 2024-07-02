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

if __name__ == "__main__":

    # Load data
    texts_df = pd.read_csv("texts.csv")
    queries_df = pd.read_csv("queries.csv")
    queries_df["query_id"] = queries_df["query_id"].astype(str)
    # Replace all positive score with 1 for simplicity
    queries_df.loc[queries_df["score"] > 0, "score"] = 1

    # Select firs q for testing
    query_id = queries_df.loc[0, "query_id"]
    query_text = queries_df.loc[0, "query_text"]
    query_text = text_preprocess(query_text)

    # Create Qrel
    qrels_df = queries_df.loc[queries_df["query_id"]==query_id]
    qrels = Qrels.from_df(
        df=qrels_df,
        q_id_col="query_id",
        doc_id_col="doc_id",
        score_col="score",
    )
    

    # Split data in chunks
    doc_ids = []
    chunk_ids = []
    chunk_texts = []

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=64,
            length_function=len,
            is_separator_regex=False,
        )
    
    for i, row in texts_df.iterrows():
        doc_id = row["doc_id"]
        doc_chunk_texts = text_splitter.split_text(row["text"])
        n_chunk_texts = len(doc_chunk_texts)
        doc_chunk_ids = [f"{doc_id}-{str(i)}" for i in range(n_chunk_texts)]

        # basic text processing
        doc_chunk_texts = [text_preprocess(chunk) for chunk in doc_chunk_texts]

        doc_ids.extend([doc_id] * n_chunk_texts)
        chunk_ids.extend(doc_chunk_ids)
        chunk_texts.extend(doc_chunk_texts)

    df = pd.DataFrame({
        "doc_id": doc_ids,
        "chunk_id": chunk_ids,
        "chunk_text": chunk_texts,
    })


    ## Sentence Transformers + Faiss
    # Create embeddings for each chunk
    k = 5
    sent_model = SentenceTransformer("all-mpnet-base-v2")
    sent_embeddings = sent_model.encode(chunk_texts)
    nb, d = sent_embeddings.shape
    index = faiss.IndexFlatL2(d)
    index.add(sent_embeddings)
    # Search
    sent_query_emb = sent_model.encode(query_text).reshape(1,-1)
    D, I = index.search(sent_query_emb, k)
    sent_df = df.loc[I[0]]
    sent_df["score"] = D[0].astype(float)
    sent_df["query_id"] = str(query_id)
    # Drop to get docs, no chunks
    sent_df = sent_df.sort_values("score", ascending=False)
    sent_df = sent_df.drop_duplicates(subset=["doc_id"], keep="first")
    sent_run = Run.from_df(
        df=sent_df,
        q_id_col="query_id",
        doc_id_col="doc_id",
        score_col="score",
        name="sent_tr",
    )


    ## RAGatuille + Colbert
    # Define index / Add option to load
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    try:
        RAG.from_index(".ragatouille/colbert/indexes/index")
    except:
        RAG.index(
            index_name="index", 
            collection=chunk_texts, 
            document_ids=chunk_ids, 
            use_faiss=False,
            max_document_length=1024,
            split_documents=False,
        )
    # Run query
    colbert_results = RAG.search(query_text)
    # Save results as a df
    colbert_df = pd.DataFrame(colbert_results)
    colbert_df = colbert_df.merge(df, how="left", left_on="document_id", right_on="chunk_id")
    colbert_df["query_id"] = str(query_id)
    # Drop to get docs, no chunks
    colbert_df = colbert_df.sort_values("score", ascending=False)
    colbert_df = colbert_df.drop_duplicates(subset=["doc_id"], keep="first")
    # Save results in Run format
    colbert_run = Run.from_df(
        df=colbert_df,
        q_id_col="query_id",
        doc_id_col="doc_id",
        score_col="score",
        name="colbert",
    )


    ## BM 25
    tokenized_corpus = [doc.split(" ") for doc in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query_text.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    bm25_df = df.copy()
    bm25_df["score"] = doc_scores
    bm25_df = bm25_df.loc[bm25_df["score"] > 0]
    bm25_df["query_id"] = str(query_id)
    # Drop to get docs, no chunks
    bm25_df = bm25_df.sort_values("score", ascending=False)
    bm25_df = bm25_df.drop_duplicates(subset=["doc_id"], keep="first")
    # Save results in Run format
    bm25_run = Run.from_df(
        df=bm25_df,
        q_id_col="query_id",
        doc_id_col="doc_id",
        score_col="score",
        name="bm25",
    )


    ## Combining scores
    combined_run = fuse(
        runs=[bm25_run, sent_run, colbert_run],
        norm="min-max",
        method="max",
    )


    ## Evaluate runs
    evaluate(qrels, combined_run, ["ndcg@5", "mrr"])
