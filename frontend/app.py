import streamlit as st
import requests
import time
import json

# Configuration
API_URL = "http://localhost:8000/chat"
API_HEALTH_URL = "http://localhost:8000/health"

# Page config
st.set_page_config(
    page_title="UET AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        animation: fadeIn 0.5s;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .info-box {
        padding: 15px;
        background-color: #fff3cd;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        padding: 15px;
        background-color: #d4edda;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    .error-box {
        padding: 15px;
        background-color: #f8d7da;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'avg_response_time' not in st.session_state:
    st.session_state.avg_response_time = 0
if 'api_status' not in st.session_state:
    st.session_state.api_status = "Unknown"

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(API_HEALTH_URL, timeout=2)
        if response.status_code == 200:
            return "✅ Online"
        return "⚠️ Degraded"
    except:
        return "❌ Offline"

def send_message(message: str):
    """Send message to the API and get response."""
    try:
        start_time = time.time()
        response = requests.post(
            API_URL,
            json={"message": message},
            timeout= 180
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "response": data.get("response", "No response received"),
                "processing_time": data.get("processing_time", elapsed)
            }
        else:
            return {
                "success": False,
                "response": f"Error: {response.status_code} - {response.text}",
                "processing_time": elapsed
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "response": "Request timed out. The server might be processing a complex query.",
            "processing_time": 30
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "response": "Cannot connect to the API. Make sure the server is running on localhost:8000",
            "processing_time": 0
        }
    except Exception as e:
        return {
            "success": False,
            "response": f"Error: {str(e)}",
            "processing_time": 0
        }

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 UET AI Assistant</h1>
    <p>Your intelligent guide to UET programs, admissions, and facilities</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI assistant helps you find information about:
    - 📚 Department programs & courses
    - 📝 Admission requirements
    - 👨‍🏫 Faculty information
    """)
    
    st.divider()
    
    # API Status
    st.header("🔧 System Status")
    if st.button("Check API Status", use_container_width=True):
        st.session_state.api_status = check_api_health()
    
    status_color = "green" if "✅" in st.session_state.api_status else "red" if "❌" in st.session_state.api_status else "orange"
    st.markdown(f"**API Status:** :{status_color}[{st.session_state.api_status}]")
    
    st.divider()
    
    # Statistics
    st.header("📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        st.metric("Avg Time", f"{st.session_state.avg_response_time:.2f}s")
    
    st.divider()
    
    # Sample questions
    st.header("💡 Sample Questions")
    sample_questions = [
        "What is the admission criteria for Computer Science?",
        "Who is the current Dean of the Faculty of Mechanical Engineering?",
        "I have completed my 16 years of education in Computer Science. Am I eligible to apply for the M.Sc. Data Science program?",
        "I see a program called M.Sc. Disaster Management and another called M.Sc. Disaster Mitigation Engineering. Which one is offered by the Civil Engineering department?",
        "I heard the Department of Petroleum & Gas Engineering is highly ranked. Is there any mention of their world ranking in the prospectus?",
        "I am looking for the faculty list for the Department of Mathematics. Who is the Chairperson?"
    ]
    
    for question in sample_questions:
        if st.button(question, key=question, use_container_width=True):
            st.session_state.user_input = question
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat area
st.header("💬 Chat")

# Display welcome message if no messages
if not st.session_state.messages:
    st.markdown("""
    <div class="info-box">
        <strong>👋 Welcome!</strong><br>
        Ask me anything about UET departments, admissions, courses, fees, or facilities.<br>
        Try using the sample questions in the sidebar to get started!
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            processing_time = message.get("processing_time", 0)
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 UET Assistant</strong> <small>({processing_time:.2f}s)</small><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask me about UET...", key="chat_input")

# Handle input from sample questions
if hasattr(st.session_state, 'user_input') and st.session_state.user_input:
    user_input = st.session_state.user_input
    delattr(st.session_state, 'user_input')

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Show thinking indicator
    with st.spinner("🤔 Thinking..."):
        result = send_message(user_input)
    
    # Update statistics
    st.session_state.total_queries += 1
    if result.get("processing_time", 0) > 0:
        # Calculate running average
        old_avg = st.session_state.avg_response_time
        n = st.session_state.total_queries
        new_time = result["processing_time"]
        st.session_state.avg_response_time = ((old_avg * (n - 1)) + new_time) / n
    
    # Add bot response
    if result["success"]:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "processing_time": result["processing_time"]
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ {result['response']}",
            "processing_time": result["processing_time"]
        })
    
    # Rerun to update chat
    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <small>
        🎓 UET AI Assistant | Powered by RAG & LLM Technology<br>
        For the most accurate information, please verify with official UET sources.
    </small>
</div>
""", unsafe_allow_html=True)