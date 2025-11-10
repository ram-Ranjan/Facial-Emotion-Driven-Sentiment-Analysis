import os, json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

os.makedirs("outputs", exist_ok=True)

reviews_path = "outputs/reviews.json"
if not os.path.exists(reviews_path):
    raise FileNotFoundError("Missing reviews.json — run synth_reviews.py first!")

reviews = json.load(open(reviews_path))
docs = [str(r) for r in reviews]

embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_texts(docs, embedding_fn)
print(f"✅ Stored {len(docs)} reviews in FAISS vector store")

user_query = "What do users feel about the product?"
query_emb = embedding_fn.embed_query(user_query)
docs_and_scores = db.similarity_search_by_vector(query_emb, k=5)

print(f"\n🔍 Query: {user_query}\nTop Retrieved Reviews:")
for doc in docs_and_scores:
    print("👉", doc.page_content)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
combined_text = " ".join([doc.page_content for doc in docs_and_scores])
summary = summarizer(combined_text, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
print("\n🧠 Summary of Retrieved Reviews:\n", summary)

sentiment = pipeline("sentiment-analysis") 
for i, txt in enumerate(docs[:5]):
    res = sentiment(txt)[0]
    print(f"\nReview {i+1}: {txt}\nSentiment → {res['label']} ({res['score']:.2f})")
