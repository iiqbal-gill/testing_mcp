import os
import re
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/vector_db")

# Initialize embeddings and vectorstore
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# Initialize LLM
llm = ChatOllama(model="gemma3", temperature=0)

# ==================== IMPROVED QUERY EXPANSION ====================
def expand_query(original_query: str) -> List[str]:
    """
    Generate multiple search queries with focus on faculty/staff information.
    """
    queries = [original_query]
    query_lower = original_query.lower()
    
    # For faculty/chairperson/dean queries - CRITICAL for your issue
    if any(word in query_lower for word in ["chairperson", "chair person", "head of department", "hod", "dean", "faculty", "professor", "lecturer"]):
        # Add specific variations
        queries.append(f"faculty list {original_query}")
        queries.append(f"staff members {original_query}")
        
        # Extract department name
        for dept in ["mathematics", "computer science", "electrical", "mechanical", "civil", 
                     "chemical", "industrial", "architecture", "city planning", "regional planning"]:
            if dept in query_lower:
                queries.append(f"{dept} department faculty")
                queries.append(f"{dept} chairperson head")
                break
    
    # For department program queries
    if any(word in query_lower for word in ["m.sc", "msc", "phd", "bachelor", "master", "program", "degree"]):
        queries.append(f"program offered by department {original_query}")
        
        # Extract program name
        for program in ["artificial intelligence", "ai", "mining engineering", "transportation", 
                       "geological engineering", "geotechnical"]:
            if program in query_lower:
                queries.append(f"{program} program department")
                break
    
    # For admission queries
    if any(word in query_lower for word in ["admission", "apply", "eligibility", "requirement"]):
        queries.append(f"admission requirements {original_query}")
        queries.append(f"eligibility criteria {original_query}")
    
    # For transportation engineering - specific fix
    if "transportation" in query_lower:
        queries.append("civil engineering transportation")
        queries.append("transportation engineering department")
    
    # For geological vs geotechnical
    if "geological" in query_lower or "geotechnical" in query_lower:
        queries.append("mining geological engineering")
        queries.append("civil geotechnical engineering")
    
    # Return unique queries, max 4
    return list(dict.fromkeys(queries))[:4]

# ==================== ENHANCED RETRIEVAL WITH RERANKING ====================
def search_prospectus(query: str) -> Dict:
    """
    Enhanced search with better relevance scoring.
    """
    print(f"   ⚡ Searching: '{query}'...")
    
    try:
        # Step 1: Expand query
        queries = expand_query(query)
        print(f"   🔍 Expanded to {len(queries)} queries: {queries}")
        
        # Step 2: Retrieve documents
        all_docs = []
        seen_content = set()
        
        for q in queries:
            # Use MMR for diversity
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,  # Get 4 per query
                    "fetch_k": 15,  # Consider 15 candidates
                    "lambda_mult": 0.5  # More diversity for faculty info
                }
            )
            docs = retriever.invoke(q)
            
            # Deduplicate
            for doc in docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        if not all_docs:
            return {
                "found": False,
                "context": "No relevant information found in the prospectus.",
                "sources": []
            }
        
        # Step 3: RERANK based on query keywords
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        scored_docs = []
        for doc in all_docs:
            content_lower = doc.page_content.lower()
            
            # Calculate relevance score
            score = 0
            
            # Exact phrase match (highest priority)
            for length in [4, 3, 2]:  # Check for 4, 3, 2 word phrases
                words = query_lower.split()
                for i in range(len(words) - length + 1):
                    phrase = " ".join(words[i:i+length])
                    if phrase in content_lower:
                        score += length * 10  # High score for phrase matches
            
            # Individual keyword matches
            doc_words = set(content_lower.split())
            keyword_matches = query_keywords.intersection(doc_words)
            score += len(keyword_matches) * 2
            
            # Boost for faculty-related terms in faculty queries
            if any(word in query_lower for word in ["chairperson", "faculty", "head", "dean", "professor"]):
                faculty_terms = ["dr.", "prof.", "chairperson", "head of department", "professor", "lecturer"]
                for term in faculty_terms:
                    if term in content_lower:
                        score += 15  # Big boost
            
            # Boost for department name match
            dept_meta = doc.metadata.get("departments", "").lower()
            for dept_word in ["mathematics", "computer", "electrical", "civil", "mechanical", "mining"]:
                if dept_word in query_lower and dept_word in (content_lower + " " + dept_meta):
                    score += 10
            
            scored_docs.append((score, doc))
        
        # Sort by score and take top documents
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        top_docs = [doc for score, doc in scored_docs[:6]]  # Top 6
        
        print(f"   📚 Retrieved {len(top_docs)} relevant documents")
        print(f"   Top scores: {[score for score, _ in scored_docs[:3]]}")
        
        # Step 4: Build context with full content
        context_parts = []
        sources = []
        
        for i, doc in enumerate(top_docs, 1):
            content = doc.page_content.strip()
            dept = doc.metadata.get("departments", "General")
            section = doc.metadata.get("section_type", "general_info")
            page = doc.metadata.get("page", "unknown")
            
            context_parts.append(
                f"[Source {i} - {dept} - Page {page}]\n{content}\n"
            )
            
            sources.append({
                "page": page,
                "department": dept,
                "section": section
            })
        
        context = "\n".join(context_parts)
        
        return {
            "found": True,
            "context": context,
            "sources": sources,
            "doc_count": len(top_docs)
        }
        
    except Exception as e:
        print(f"   ❌ Search error: {str(e)}")
        return {
            "found": False,
            "context": f"Search error: {str(e)}",
            "sources": []
        }

# ==================== FIXED AGENT - ACTUALLY EXECUTES SEARCH ====================
def run_enhanced_agent(user_query: str) -> str:
    """
    Fixed agent that actually executes searches instead of returning "Action: Search".
    """
    
    # Check if this is clearly a UET factual question - if so, skip decision step
    query_lower = user_query.lower()
    
    # Direct search triggers
    factual_indicators = [
        "which department", "what department", "who is", "is there", "tell me about",
        "faculty", "chairperson", "dean", "head of department", "professor",
        "m.sc", "msc", "phd", "bachelor", "master", "program", "offers",
        "admission", "eligibility", "requirement", "apply",
        "how many", "list of", "what are"
    ]
    
    should_search_directly = any(indicator in query_lower for indicator in factual_indicators)
    
    if should_search_directly:
        print("🎯 Direct search triggered (factual query detected)")
        search_results = search_prospectus(user_query)
        
        if not search_results["found"]:
            return "I couldn't find specific information about that in the UET prospectus. Could you rephrase your question or ask about a specific department?"
        
        # Generate answer with explicit instructions
        answer_prompt = f"""You are answering a question about UET based on the official prospectus.

USER QUESTION: {user_query}

CONTEXT FROM PROSPECTUS:
{search_results["context"]}

CRITICAL INSTRUCTIONS:
1. Answer the question DIRECTLY using ONLY the information in the context above.
2. If asked about faculty, chairperson, or dean, give the EXACT names from the context.
3. If asked about which department offers a program, state the EXACT department name from the context.
4. Do NOT confuse different departments - pay attention to department names in the context.
5. Be specific with names, departments, and details.
6. If the context mentions multiple departments, make it clear which information belongs to which department.
7. Keep your answer focused and accurate.

ANSWER:"""

        answer = llm.invoke(answer_prompt).content.strip()
        return answer
    
    # For non-factual queries, use decision step
    system_prompt = f"""You are the UET Prospectus AI Assistant.

USER QUERY: {user_query}

If this is a greeting, respond naturally.
If this asks for UET information, output: Action: Search [your search query]

Decision:"""

    response = llm.invoke(system_prompt).content
    print(f"🤖 Agent Decision: {response[:100]}...")
    
    # Parse for search action
    match = re.search(r"Action:\s*Search\s*\[(.+?)\]", response, re.IGNORECASE | re.DOTALL)
    
    if match:
        search_query = match.group(1).strip().strip('"').strip("'")
        print(f"🔎 Executing search: '{search_query}'")
        
        search_results = search_prospectus(search_query)
        
        if not search_results["found"]:
            return "I couldn't find specific information about that in the UET prospectus. Could you rephrase your question?"
        
        answer_prompt = f"""Based on the UET prospectus, answer this question accurately:

USER QUESTION: {user_query}

CONTEXT:
{search_results["context"]}

Give a direct, accurate answer using only the information provided. Be specific with names and departments.

ANSWER:"""

        return llm.invoke(answer_prompt).content.strip()
    
    # If refused, force search anyway
    response_lower = response.lower()
    if any(phrase in response_lower for phrase in ["don't have", "cannot", "not sure"]):
        print("⚠️ LLM refused, forcing search...")
        search_results = search_prospectus(user_query)
        
        if search_results["found"]:
            return llm.invoke(f"Answer based on this context: {search_results['context']}\n\nQuestion: {user_query}").content.strip()
    
    return response

# ==================== GUARDRAILS ====================
def is_uet_related(query: str) -> bool:
    """Enhanced guardrail."""
    query_lower = query.lower()
    
    # Allow greetings
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good evening", "thanks", "thank you"]
    if any(query_lower.strip().startswith(g) for g in greetings) and len(query_lower.split()) <= 3:
        return True
    
    # Comprehensive UET keywords
    uet_keywords = [
        "uet", "university of engineering", "lahore", "taxila", "faisalabad",
        "department", "computer", "electrical", "mechanical", "civil", "chemical",
        "industrial", "architecture", "petroleum", "biomedical", "environmental",
        "textile", "metallurgy", "mining", "aerospace", "mathematics", "physics",
        "transportation", "geological", "geotechnical", "city planning", "regional planning",
        "course", "admission", "faculty", "fee", "program", "degree", "bachelor",
        "master", "phd", "msc", "bs", "engineering", "syllabus", "curriculum",
        "professor", "chairperson", "dean", "head of department", "hod",
        "scholarship", "campus", "hostel", "library", "artificial intelligence",
    ]
    
    return any(keyword in query_lower for keyword in uet_keywords)

# ==================== MAIN ENTRY POINT ====================
def process_query(user_query: str) -> str:
    """Main entry point with improved handling."""
    user_query = user_query.strip()
    
    if not user_query:
        return "Please ask me a question about UET."
    
    if not is_uet_related(user_query):
        return "I'm specialized in answering questions about UET departments, programs, admissions, and facilities. Please ask me about these topics!"
    
    try:
        return run_enhanced_agent(user_query)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"I encountered an error while processing your question. Please try rephrasing."

# ==================== TESTING ====================
def test_problematic_cases():
    """Test the specific cases that were failing."""
    test_cases = [
        "Can you tell me which department offers the M.Sc. Mining Engineering program?",
        "Is there any specific department that deals with Transportation Engineering?",
        "I am looking for the faculty list for the Department of Mathematics. Who is the Chairperson?",
        "I want to apply for M.Sc. Artificial Intelligence. Should I select the Department of Computer Science?",
        "I am interested in Geological Engineering. Is that the same as Geotechnical Engineering?",
    ]
    
    print("\n" + "="*80)
    print("TESTING PROBLEMATIC CASES")
    print("="*80)
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"Test {i}: {query}")
        print(f"{'─'*80}")
        response = process_query(query)
        print(f"Response: {response}")
        print(f"{'─'*80}")

if __name__ == "__main__":
    test_problematic_cases()