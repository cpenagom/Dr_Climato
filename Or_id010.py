from langchain_ollama.llms import OllamaLLM # type: ignore
from langchain_chroma import Chroma # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from langchain.schema import Document # type: ignore
import json
import os
from langchain_ollama.embeddings import OllamaEmbeddings # type: ignore

# === CONFIG ===
PERSIST_NEWS = "./db_news"
PERSIST_FACTS = "./db_factchecks"

# === Load Embeddings ===
def get_embedder():
    return OllamaEmbeddings(model="mxbai-embed-large")

# === Load JSON Data === do the chunks of the text, this is just a json
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return [
        Document(page_content=item["content"], metadata={"title": item["title"], "source": item["source"]})
        for item in data
    ]

# === Create or Load ChromaDB === DATABASE
def build_or_load_vectorstore(docs, persist_dir):
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=get_embedder())
    else:
        return Chroma.from_documents(docs, get_embedder(), persist_directory=persist_dir)

# === Load Ollama ===
def get_llm():
    return OllamaLLM(model="llama3.2")

# === Climate Relevance Classifier ===
def is_climate_related(query, llm):
    prompt = f"""Is the following query related to climate change, environmental issues, or energy policy? 
Answer only 'YES' or 'NO'.

Query: {query}

Answer:"""
    
    response = llm.invoke(prompt).strip().upper()
    print(f"[DEBUG] Climate relevance check: '{query}' -> {response}")
    return response.startswith("YES")

# === Check if retrieved news matches user query ===
def is_news_relevant(user_query, news_doc, llm):
    prompt = f"""Does the following news article match what the user is asking about?

User query: "{user_query}"
News title: "{news_doc.metadata.get('title', 'Unknown')}"
News content: "{news_doc.page_content[:200]}..."

Answer only 'YES' if the news is clearly related to what the user is asking about, or 'NO' if it's not related.

Answer:"""
    
    response = llm.invoke(prompt).strip().upper()
    print(f"[DEBUG] News relevance check: {response}")
    return response.startswith("YES")

# === CLIMATE AGENT ===
def climate_factcheck_agent(user_question, news_doc, factcheck_matches, llm):
    news_text = news_doc.page_content[:300]
    fact_checks_text = "\n\n".join(
        [f"- {fc.metadata.get('title', 'Unknown')}: {fc.page_content[:300]}" for fc in factcheck_matches]
    )

    prompt = f"""
You are Dr. Climato, an expert in climate misinformation.

A user has asked: "{user_question}"

You found the following news:
"{news_text}"

And the following fact-checks:
{fact_checks_text}

Your job is to explain clearly if the news has been fact-checked based on the provided data.
Be concise but informative and focus only on what the data shows.

Your answer:
"""
    return llm.invoke(prompt).strip()

# THE IDEA
# 1. Use CheckClimateRelevance first - if not climate-related, politely decline
# 2. Use AnalyzeQuerySpecificity - if too vague, ask for specifics
# 3. Use SearchNews to find relevant news articles
# 4. Use SearchFactChecks to find relevant fact-checks
# 5. ALWAYS provide a comprehensive final answer that includes:
#    - A direct answer to the user's question
#    - Summary of findings from news articles (if found)
#    - Summary of findings from fact-checks (if found)
#    - Clear conclusion based on the evidence

# === Enhanced Pipeline with Proper Checks ===
def dual_rag_pipeline(query, llm, news_db, fact_db):
    # Step 1: Check if query is climate-related
    if not is_climate_related(query, llm):
        return "❌ Sorry, this is not climate change related. I can only help with climate change and environmental topics.", []
    
    # Step 2: Retrieve matching news articles
    news_matches = news_db.similarity_search(query, k=3)  # Get top 3 to check relevance
    if not news_matches:
        return "❌ Sorry, I couldn't find any relevant news in my database for your climate-related query.", []

    # Step 3: Check if any retrieved news is actually relevant
    relevant_news = None
    for news in news_matches:
        print(f"[DEBUG] Checking news: {news.metadata.get('title', 'Unknown')}")
        if is_news_relevant(query, news, llm):
            relevant_news = news
            break
    
    if not relevant_news:
        return "❌ Sorry, I couldn't find any news in my database that matches your specific query about climate change.", []

    print(f"[DEBUG] Selected relevant news: {relevant_news.metadata.get('title', 'Unknown')}")

    # Step 4: Use news content to retrieve factchecks
    fact_matches = fact_db.similarity_search(relevant_news.page_content, k=3)
    
    # Step 5: Check if we have meaningful fact-checks
    if not fact_matches:
        return "❌ Sorry, I found relevant news but no fact-checks available in my database for this topic.", []
    
    print(f"[DEBUG] Found {len(fact_matches)} fact-checks")

    # Step 6: Pass through reasoning agent only if we have both news and fact-checks
    result = climate_factcheck_agent(query, relevant_news, fact_matches, llm)
    sources = [f"{doc.metadata['title']} — {doc.metadata['source']}" for doc in fact_matches if doc.metadata]

    return result, sources

# === Main CLI ===
if __name__ == "__main__":
    print("🔍 Initializing climate fact-checking agent...")

    # Load and embed data
    news_docs = load_json("data/news.json")
    fact_docs = load_json("data/factchecks.json")

    news_db = build_or_load_vectorstore(news_docs, PERSIST_NEWS)
    fact_db = build_or_load_vectorstore(fact_docs, PERSIST_FACTS)

    llm = get_llm()
    
    print("🌍 Welcome to the Climate FactCheck CLI. Type your query (type 'exit' to quit):")

    while True:
        query = input("\n🧾 You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break  

        if not query:  # Handle empty input
            continue

        result, sources = dual_rag_pipeline(query, llm, news_db, fact_db)
        
        print("\n✅", result)
        if sources:
            print("📚 Sources:")
            for src in sources:
                print(" -", src)

# PROVIDE AND PARSER OR SOMETHING FOR STANDS ANALYSIS
