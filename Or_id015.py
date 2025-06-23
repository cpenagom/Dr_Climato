# import warnings
# # Suppress all LangChain deprecation warnings
# warnings.filterwarnings("ignore", message=".*LangChain.*")
# warnings.filterwarnings("ignore", message=".*Please see the migration guide.*")

from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema import Document
import json
import os
from langchain_ollama.embeddings import OllamaEmbeddings 
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
import re

# === CONFIG ===
PERSIST_NEWS = "./db_news"
PERSIST_FACTS = "./db_factchecks"

# === Load Embeddings ===
def get_embedder():
    return OllamaEmbeddings(model="mxbai-embed-large")

# === Load JSON Data ===
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return [
        Document(page_content=item["content"], metadata={"title": item["title"], "source": item["source"]})
        for item in data
    ]

# === Create or Load ChromaDB ===
def build_or_load_vectorstore(docs, persist_dir):
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=get_embedder())
    else:
        return Chroma.from_documents(docs, get_embedder(), persist_directory=persist_dir)

# === Load Ollama with better parameters for verbose output ===
def get_llm():
    return OllamaLLM(
        model="llama3.2",
        temperature=0.1,  # Lower temperature for more consistent responses
        num_predict=512,  # Increased to allow for detailed reasoning
        stop=["\nHuman:", "\nUser:"]  # Stop at human input, but allow reasoning chains
    )

# === Improved Climate Relevance Classifier ===
def is_climate_related(query, llm):
    prompt = f"""You are a climate topic classifier. Determine if the following query is related to 
    climate change, environmental issues, global warming, carbon emissions, renewable energy, 
    or environmental policy.

Query: "{query}"

Respond with exactly one word: YES or NO

Response:"""
    
    try:
        response = llm.invoke(prompt).strip().upper()
        # Extract just YES or NO from the response
        if "YES" in response:
            result = "YES"
        elif "NO" in response:
            result = "NO"
        else:
            result = "NO"  # Default to NO if unclear
        
        print(f"[DEBUG] Climate relevance check: '{query}' -> {result}")
        return result == "YES"
    except Exception as e:
        print(f"[DEBUG] Error in climate check: {e}")
        return False

# === LangChain Agent Tools ===
def build_factcheck_tools(news_db, fact_db, llm):
    
    def climate_relevance_tool(query: str) -> str:
        """Check if a query is related to climate change"""
        try:
            if is_climate_related(query, llm):
                return "CLIMATE_RELATED: YES - This query is about climate change"
            else:
                return "CLIMATE_RELATED: NO - This query is not about climate change"
        except Exception as e:
            return f"CLIMATE_RELATED: ERROR - {str(e)}"
    
    def search_news_tool(query: str) -> str:
        """Search for climate-related news articles with relevance threshold"""
        try:
            # Get more results to filter through
            results = news_db.similarity_search_with_score(query, k=5)
            if not results:
                return "NEWS_SEARCH: No articles found in database"
            
            # Filter results by relevance score (lower score = more similar)
            relevant_results = []
            for doc, score in results:
                # Only include results with good similarity (score < 0.7 is a reasonable threshold)
                if score < 0.7:
                    relevant_results.append((doc, score))
            
            if not relevant_results:
                return f"NEWS_SEARCH: No articles found that specifically match '{query}'. Try being more specific or use different keywords."
            
            # Further filter by content relevance
            truly_relevant = []
            for doc, score in relevant_results[:3]:  # Check top 3
                title = doc.metadata.get('title', '')
                content = doc.page_content[:200]
                
                # Check if query keywords appear in title or content
                query_words = query.lower().split()
                title_lower = title.lower()
                content_lower = content.lower()
                
                # Count keyword matches
                matches = sum(1 for word in query_words if word in title_lower or word in content_lower)
                
                # Only include if at least 30% of query words match
                if matches >= len(query_words) * 0.3:
                    truly_relevant.append((doc, score, matches))
            
            if not truly_relevant:
                return f"NEWS_SEARCH: Found articles about climate change, but none specifically about '{query}'. Please be more specific."
            
            # Sort by number of matches and similarity score
            truly_relevant.sort(key=lambda x: (-x[2], x[1]))
            
            output = [f"NEWS_SEARCH: Found {len(truly_relevant)} article(s) specifically about '{query}':"]
            for i, (doc, score, matches) in enumerate(truly_relevant, 1):
                title = doc.metadata.get('title', 'Unknown')
                content = doc.page_content[:150].replace('\n', ' ')
                output.append(f"Article {i}: {title} - {content}... (Relevance: {matches} keyword matches)")
            
            return "\n".join(output)
        except Exception as e:
            return f"NEWS_SEARCH: Error - {str(e)}"

    def search_facts_tool(query: str) -> str:
        """Search for fact-checks related to climate topics with fixed thresholds"""
        try:
            # Get results with similarity scores
            results = fact_db.similarity_search_with_score(query, k=5)
            if not results:
                return "FACT_CHECK: No fact-checks found in database"
            
            # FIXED: More lenient similarity threshold - your fact-check has score 0.6098
            # Lower score = MORE similar, so we want scores BELOW a threshold
            relevant_results = []
            for doc, score in results:
                if score < 0.8:  # Accept scores below 0.7 (your 0.6098 will pass)
                    relevant_results.append((doc, score))
            
            if not relevant_results:
                return f"FACT_CHECK: No fact-checks found that relate to '{query}'. Try different keywords."
            
            # Improved content relevance checking
            truly_relevant = []
            for doc, score in relevant_results[:4]:  # Check top 4
                title = doc.metadata.get('title', '')
                content = doc.page_content
                
                # Extract meaningful keywords from query (exclude common words)
                stop_words = {'can', 'you', 'explain', 'me', 'if', 'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'this', 'that', 'what', 'how', 'why', 'when', 'where', 'about', 'than', 'faster', 'slower', 'more', 'less', 'growing', 'increasing', 'decreasing'}
                
                query_words = [word.lower().strip() for word in query.split() if word.lower().strip() not in stop_words and len(word.strip()) > 2]
                
                # Key concepts with synonyms
                concept_map = {
                    'solar': ['solar', 'photovoltaic', 'pv', 'sun'],
                    'wind': ['wind', 'turbine', 'windmill'],
                    'power': ['power', 'energy', 'electricity', 'generation'],
                    'nuclear': ['nuclear', 'atomic', 'reactor'],
                    'fossil': ['fossil', 'coal', 'oil', 'gas', 'petroleum'],
                    'renewable': ['renewable', 'clean', 'green', 'sustainable'],
                    'climate': ['climate', 'warming', 'temperature', 'emissions', 'carbon']
                }
                
                title_lower = title.lower()
                content_lower = content.lower()
                
                # Count matches including synonyms
                matches = 0
                matched_concepts = []
                
                for word in query_words:
                    # Direct word match
                    if word in title_lower or word in content_lower:
                        matches += 1
                        matched_concepts.append(word)
                    else:
                        # Check for concept matches
                        for concept, synonyms in concept_map.items():
                            if word in synonyms:
                                for synonym in synonyms:
                                    if synonym in title_lower or synonym in content_lower:
                                        matches += 1
                                        matched_concepts.append(f"{word}({synonym})")
                                        break
                                break
                
                # FIXED: Require just 1 match OR very good similarity
                min_matches = 1  # Just need 1 keyword match
                very_good_similarity = score < 0.6  # Very similar content
                
                if matches >= min_matches or very_good_similarity:
                    truly_relevant.append((doc, score, matches, matched_concepts))
            
            if not truly_relevant:
                # FIXED: Show results anyway if we have good similarity scores
                fallback_results = relevant_results[:3]  # Show top 3 by similarity
                output = [f"FACT_CHECK: Found related fact-checks for '{query}':"]
                for i, (doc, score) in enumerate(fallback_results, 1):
                    title = doc.metadata.get('title', 'Unknown')
                    source = doc.metadata.get('source', 'Unknown')
                    content = doc.page_content[:150].replace('\n', ' ')
                    output.append(f"Related fact-check {i}: {title} ({source}) - {content}... (Similarity: {score:.3f})")
                return "\n".join(output)
            
            # Sort by number of matches first, then by similarity score
            truly_relevant.sort(key=lambda x: (-x[2], x[1]))
            
            output = [f"FACT_CHECK: Found {len(truly_relevant)} relevant fact-check(s) for '{query}':"]
            for i, (doc, score, matches, concepts) in enumerate(truly_relevant, 1):
                title = doc.metadata.get('title', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content[:200].replace('\n', ' ')
                
                if concepts:
                    concepts_str = ', '.join(concepts[:3])  # Show first 3 matched concepts
                    match_info = f"Matched: {concepts_str}"
                else:
                    match_info = f"Similar content (score: {score:.3f})"
                
                output.append(f"Fact-check {i}: {title} ({source}) - {content}... ({match_info})")
            
            return "\n".join(output)
        except Exception as e:
            return f"FACT_CHECK: Error - {str(e)}"

    
    def query_specificity_tool(query: str) -> str:
        """Analyze if the query is specific enough to find targeted results"""
        try:
            # Check if query is too vague
            vague_patterns = [
                "what about", "tell me about", "climate change", "global warming",
                "environment", "what is", "how is", "anything about"
            ]
            
            query_lower = query.lower().strip()
            
            # Check for vague patterns
            is_vague = any(pattern in query_lower for pattern in vague_patterns)
            
            # Check if query has specific keywords
            specific_keywords = [
                "study", "research", "report", "data", "statistics", "temperature",
                "emissions", "carbon", "renewable", "solar", "wind", "fossil",
                "ice", "ocean", "forest", "policy", "agreement", "protocol"
            ]
            
            has_specific_terms = any(keyword in query_lower for keyword in specific_keywords)
            
            # Count words (excluding common words)
            common_words = {"about", "what", "how", "is", "the", "and", "or", "of", "to", "in", "a", "an"}
            meaningful_words = [word for word in query_lower.split() if word not in common_words]
            
            if is_vague and not has_specific_terms and len(meaningful_words) < 3:
                return f"QUERY_ANALYSIS: Too vague - '{query}' needs more specific details (e.g., specific studies, policies, or phenomena)"
            elif has_specific_terms and len(meaningful_words) >= 2:
                return f"QUERY_ANALYSIS: Specific enough - '{query}' should return targeted results"
            else:
                return f"QUERY_ANALYSIS: Moderately specific - '{query}' might return broad results"
                
        except Exception as e:
            return f"QUERY_ANALYSIS: Error - {str(e)}"

    tools = [
        Tool(
            name="CheckClimateRelevance",
            func=climate_relevance_tool,
            description="Check if a query is related to climate change. Use this FIRST for every query."
        ),
        Tool(
            name="AnalyzeQuerySpecificity",
            func=query_specificity_tool,
            description="Check if the user's query is specific enough to find targeted results. Use this SECOND after confirming climate relevance."
        ),
        Tool(
            name="SearchNews",
            func=search_news_tool,
            description="Search for news articles that specifically match the user's query. Only use after confirming query is climate-related. Returns 'No match' if no specific articles found."
        ),
        Tool(
            name="SearchFactChecks",
            func=search_facts_tool,
            description="Search for fact-checks that specifically address the user's query. Returns 'No match' if no relevant fact-checks found."
        )
    ]
    return tools

# === Improved Agent Creation with Better Prompting ===
def create_climate_agent(llm, tools):
    # Enhanced prompt that ensures the agent provides complete responses
    agent_prompt = """You are Dr. Climato, a climate fact-checking expert. You MUST always provide a complete, helpful response to the user.

MANDATORY PROCESS:
1. Use CheckClimateRelevance first - if not climate-related, politely decline
2. Use AnalyzeQuerySpecificity - if too vague, ask for specifics
3. Use SearchNews to find relevant news articles
4. Use SearchFactChecks to find relevant fact-checks
5. ALWAYS provide a comprehensive final answer that includes:
   - A direct answer to the user's question
   - Summary of findings from news articles (if found)
   - Summary of findings from fact-checks (if found)
   - Clear conclusion based on the evidence

CRITICAL RULES:
- NEVER just list tool results - always synthesize them into a human-readable response
- If you find fact-checks, ALWAYS explain what they say about the user's question
- If you find news articles, ALWAYS summarize their key points
- Always end with a clear, direct answer to the user's original question
- If no specific matches found, say so clearly but offer related information

RESPONSE FORMAT:
Start with: "Based on my search of climate databases..."
Include: What the evidence shows about their specific question
End with: A clear conclusion or recommendation

Remember: You are here to help users understand climate topics with accurate information."""

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,  # Allow more iterations for complete responses
        early_stopping_method="generate",
        agent_kwargs={
            'prefix': agent_prompt,
            'format_instructions': """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [CheckClimateRelevance, AnalyzeQuerySpecificity, SearchNews, SearchFactChecks]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer and will provide a complete response
Final Answer: A comprehensive response that directly addresses the user's question with evidence from my searches"""
        }
    )
    return agent

# === Enhanced Fallback Processing ===
def fallback_processing(query, tools):
    """Enhanced fallback that ensures complete responses"""
    try:
        print(f"\n🤖 ENHANCED FALLBACK PROCESSING: '{query}'")
        print("-" * 50)
        
        results = {}
        
        # Step 1: Check climate relevance
        print("💭 Checking climate relevance...")
        relevance_tool = tools[0]
        results['relevance'] = relevance_tool.func(query)
        print(f"Result: {results['relevance']}")
        
        if "NO" in results['relevance']:
            return "I specialize in climate change and environmental topics. Please ask me about climate-related issues like global warming, renewable energy, emissions, or environmental policies."
        
        # Step 2: Check specificity
        print("\n💭 Analyzing query specificity...")
        specificity_tool = tools[1]
        results['specificity'] = specificity_tool.func(query)
        print(f"Result: {results['specificity']}")
        
        if "Too vague" in results['specificity']:
            return f"Your question needs more specifics. {results['specificity'].split('(e.g.,')[1].rstrip(')') if '(e.g.,' in results['specificity'] else 'Please be more specific about what aspect of climate change you want to know about.'}"
        
        # Step 3: Search news
        print("\n💭 Searching for news articles...")
        news_tool = tools[2]
        results['news'] = news_tool.func(query)
        print(f"Result: {results['news'][:200]}...")
        
        # Step 4: Search fact-checks
        print("\n💭 Searching for fact-checks...")
        fact_tool = tools[3]
        results['facts'] = fact_tool.func(query)
        print(f"Result: {results['facts'][:200]}...")
        
        # Step 5: Generate comprehensive response
        print("\n💭 Generating comprehensive response...")
        
        response_parts = []
        response_parts.append("Based on my search of climate databases:")
        
        # Add news findings
        if "Found" in results['news'] and "specifically" in results['news']:
            response_parts.append(f"\n📰 **News Articles Found:**")
            # Extract key information from news results
            news_lines = results['news'].split('\n')[1:]  # Skip the "Found X articles" line
            for line in news_lines[:2]:  # Show first 2 articles
                if line.strip():
                    response_parts.append(f"• {line.strip()}")
        
        # Add fact-check findings
        if "Found" in results['facts'] and "relevant fact-check" in results['facts']:
            response_parts.append(f"\n✅ **Fact-Checks Found:**")
            # Extract key information from fact-check results
            fact_lines = results['facts'].split('\n')[1:]  # Skip the "Found X fact-checks" line
            for line in fact_lines[:2]:  # Show first 2 fact-checks
                if line.strip():
                    response_parts.append(f"• {line.strip()}")
        
        # Add conclusion
        if len(response_parts) > 1:  # We found something
            response_parts.append(f"\n🎯 **Regarding your question about '{query}':**")
            
            # Try to provide a direct answer based on what we found
            if "solar" in query.lower() and "wind" in query.lower():
                if "solar" in results['facts'].lower() or "wind" in results['facts'].lower():
                    response_parts.append("The fact-checks and articles above provide evidence about the relative growth of solar vs wind power. Check the specific details in the sources listed.")
                else:
                    response_parts.append("Both solar and wind power are growing rapidly, but the specific comparison depends on the timeframe and region. The sources above provide current information on this topic.")
            else:
                response_parts.append("The sources above provide current information relevant to your climate question.")
        else:
            response_parts.append(f"\nI couldn't find specific information about '{query}' in my current databases. Try rephrasing your question with different keywords or ask about a related climate topic.")
        
        final_response = "\n".join(response_parts)
        print(f"\n✅ Generated response: {final_response[:200]}...")
        
        return final_response
        
    except Exception as e:
        return f"I encountered an error while processing your climate question: {str(e)}. Please try rephrasing your question."

# === Enhanced Safe Agent Run ===
def safe_agent_run(agent, query, max_retries=2):
    """Enhanced agent runner that ensures complete responses"""
    
    for attempt in range(max_retries + 1):
        try:
            print(f"\n{'='*60}")
            print(f"🔍 PROCESSING QUERY (Attempt {attempt + 1}): '{query}'")
            print(f"{'='*60}")
            
            # Run agent
            result = agent.invoke(query)
            
            # Extract the actual response
            if isinstance(result, dict) and 'output' in result:
                final_answer = result['output']
            elif isinstance(result, str):
                final_answer = result
            else:
                final_answer = str(result)
            
            print(f"\n{'='*60}")
            print("🎯 RAW AGENT RESULT:")
            print(f"{'='*60}")
            print(final_answer)
            
            # Check if response is complete
            if len(final_answer.strip()) < 50:  # Too short
                print("\n⚠️ Response too short, trying fallback...")
                raise Exception("Response too brief")
            
            if "FACT_CHECK: Found" in final_answer and "Based on" not in final_answer:
                print("\n⚠️ Response contains raw tool output, trying fallback...")
                raise Exception("Raw tool output in response")
            
            # Clean up response
            final_answer = re.sub(r'```[^`]*```', '', final_answer)
            final_answer = final_answer.strip()
            
            return final_answer
            
        except Exception as e:
            print(f"\n❌ [Attempt {attempt + 1}] Error: {str(e)}")
            
            if attempt == max_retries:
                print(f"\n{'='*60}")
                print("🔄 ENHANCED FALLBACK MODE")
                print(f"{'='*60}")
                return fallback_processing(query, agent.tools)
            
            print("🔄 Retrying...")
    
    return "I apologize, but I'm having trouble processing your request. Please try rephrasing your question."

## DEBUGIN why i do not find the factcheks!!!
def debug_factcheck_search(query: str, fact_db) -> str:
    """Debug the fact-check search to see what's happening"""
    try:
        print(f"\n🔍 DEBUGGING FACT-CHECK SEARCH FOR: '{query}'")
        print("=" * 60)
        
        # Step 1: Check if database has any documents
        print("📊 Step 1: Database Statistics")
        try:
            # Get all documents (this might be slow for large DBs)
            all_results = fact_db.similarity_search("", k=100)  # Get many results
            print(f"Total documents in fact-check DB: {len(all_results)}")
            
            # Show first few documents
            print("\n📝 Sample documents in database:")
            for i, doc in enumerate(all_results[:3]):
                title = doc.metadata.get('title', 'No title')
                content = doc.page_content[:100]
                print(f"  {i+1}. Title: {title}")
                print(f"     Content: {content}...")
                print(f"     Metadata: {doc.metadata}")
                print()
        except Exception as e:
            print(f"Error accessing database: {e}")
            return f"Database access error: {e}"
        
        # Step 2: Test similarity search with different queries
        print("🔍 Step 2: Similarity Search Tests")
        test_queries = [
            query,  # Original query
            "solar power",  # Simple version
            "wind power",   # Alternative
            "solar",        # Single word
            "wind",         # Single word
            "power",        # Generic
            "energy"        # Generic
        ]
        
        for test_query in test_queries:
            print(f"\nTesting query: '{test_query}'")
            results = fact_db.similarity_search_with_score(test_query, k=3)
            print(f"  Found {len(results)} results")
            
            for i, (doc, score) in enumerate(results):
                title = doc.metadata.get('title', 'No title')
                content = doc.page_content[:80]
                print(f"    {i+1}. Score: {score:.4f} | Title: {title}")
                print(f"        Content: {content}...")
        
        # Step 3: Check your specific fact-check
        print("\n🎯 Step 3: Looking for your specific fact-check")
        target_title = "Wind power becoming dominant"
        target_content = "Claims about solar is dominance are not real"
        
        found_target = False
        for doc in all_results:
            if target_title in doc.metadata.get('title', '') or target_content in doc.page_content:
                print(f"✅ Found target fact-check!")
                print(f"   Title: {doc.metadata.get('title', 'No title')}")
                print(f"   Content: {doc.page_content}")
                print(f"   Metadata: {doc.metadata}")
                found_target = True
                
                # Test similarity with this specific document
                print(f"\n🔬 Testing similarity with target document:")
                for test_query in ["solar power", "wind power", query]:
                    results = fact_db.similarity_search_with_score(test_query, k=10)
                    for i, (result_doc, score) in enumerate(results):
                        if result_doc.page_content == doc.page_content:
                            print(f"   Query '{test_query}' -> Rank {i+1}, Score: {score:.4f}")
                            break
                    else:
                        print(f"   Query '{test_query}' -> Not in top 10 results")
                break
        
        if not found_target:
            print("❌ Target fact-check not found in database!")
            print("This suggests the document wasn't properly loaded or indexed.")
        
        # Step 4: Check embedding model
        print(f"\n🤖 Step 4: Embedding Model Check")
        try:
            embedder = fact_db._embedding_function
            print(f"Embedding model: {embedder}")
            
            # Test embeddings
            query_embedding = embedder.embed_query("solar power")
            print(f"Query embedding length: {len(query_embedding) if query_embedding else 'None'}")
            
        except Exception as e:
            print(f"Embedding model error: {e}")
        
        return "Debug complete - check output above"
        
    except Exception as e:
        return f"Debug error: {str(e)}"

def simple_factcheck_search(query: str, fact_db) -> str:
    """Simplified search that shows everything"""
    try:
        print(f"\n🔍 SIMPLE SEARCH FOR: '{query}'")
        
        # Get top 10 results with scores
        results = fact_db.similarity_search_with_score(query, k=10)
        
        if not results:
            return "No results found at all"
        
        print(f"Found {len(results)} results:")
        for i, (doc, score) in enumerate(results, 1):
            title = doc.metadata.get('title', 'No title')
            content = doc.page_content[:150]
            print(f"\n{i}. SCORE: {score:.4f}")
            print(f"   TITLE: {title}")
            print(f"   CONTENT: {content}...")
            print(f"   METADATA: {doc.metadata}")
        
        return f"Found {len(results)} results (see details above)"
        
    except Exception as e:
        return f"Simple search error: {str(e)}"

# Add this to your main code to test
def test_factcheck_database():
    """Test function to run the debugger"""
    print("🔍 Loading fact-check database for testing...")
    
    # Load your database (adjust path as needed)
    fact_docs = load_json("data/factchecks.json")
    fact_db = build_or_load_vectorstore(fact_docs, PERSIST_FACTS)
    
    # Test queries
    test_query = "Can you explain me if the Solar power is growing faster than wind power?"
    
    # Run debugger
    debug_result = debug_factcheck_search(test_query, fact_db)
    print(f"\nDebug result: {debug_result}")
    
    # Run simple search
    simple_result = simple_factcheck_search(test_query, fact_db)
    print(f"\nSimple search result: {simple_result}")

# === Main CLI ===
if __name__ == "__main__":
    print("🔍 Initializing climate fact-checking agent...")
    # Add this line before your main agent loop to debug
    test_factcheck_database() # nice to see

    try:
        # Load and embed data
        print("📰 Loading news data...")
        news_docs = load_json("data/news.json")
        
        print("✅ Loading fact-check data...")
        fact_docs = load_json("data/factchecks.json")

        print("🗄️ Building vector databases...")
        news_db = build_or_load_vectorstore(news_docs, PERSIST_NEWS)
        fact_db = build_or_load_vectorstore(fact_docs, PERSIST_FACTS)

        print("🤖 Initializing language model...")
        llm = get_llm()
        
        print("🔧 Setting up tools...")
        tools = build_factcheck_tools(news_db, fact_db, llm)

        print("👨‍🔬 Creating climate agent...")
        agent = create_climate_agent(llm, tools)
        
        print("\n🌍 Welcome to Dr. Climato - Climate FactCheck Agent!")
        print("Type your climate-related questions (type 'exit' to quit):")
        print("=" * 50)

        while True:
            query = input("\n🧾 You: ").strip()
            
            if query.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break  

            if not query:  # Handle empty input
                print("Please enter a question about climate change.")
                continue

            # Use enhanced error handling - this preserves the agent's reasoning chain
            result = safe_agent_run(agent, query)
            
            # Display the result with clear formatting
            if result and not result.startswith("Sorry"):
                print(f"\n✅ Dr. Climato: {result}")
            else:
                print(f"\n⚠️ Dr. Climato: {result}")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Initialization Error: {str(e)}")
        print("Please check that your data files exist and Ollama is running.")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Initialization Error: {str(e)}")
        print("Please check that your data files exist and Ollama is running.")




    
