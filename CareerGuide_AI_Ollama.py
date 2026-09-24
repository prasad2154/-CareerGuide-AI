"""
CareerGuide AI & Document RAG Platform
- Module 1: 🎯 CareerGuide AI (Resume & Job Description Compatibility Matcher + Career Coach Chatbot)
- Module 2: 📚 Document RAG Assistant (PDF upload, vector search with ChromaDB & local Ollama embeddings, RAG QA)
"""

import os
import json
import re
import streamlit as st
from dotenv import load_dotenv
import ollama
import chromadb
from PyPDF2 import PdfReader

# Try importing python-docx if installed
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ---------------------------------------------------------
# Page & UI Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CareerGuide AI & Document RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Inject Modern Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        padding: 1.75rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.25), 0 8px 10px -6px rgba(67, 56, 202, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        background: linear-gradient(to right, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        color: #e0e7ff;
        margin-top: 0.4rem;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* Modern Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(226, 232, 240, 0.15);
        border-radius: 14px;
        padding: 1.1rem;
        transition: all 0.2s ease-in-out;
        backdrop-filter: blur(10px);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Compatibility Score Hero Box */
    .score-hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 2px solid #6366f1;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        box-shadow: 0 12px 28px rgba(99, 102, 241, 0.2);
    }
    
    .score-number {
        font-size: 3.8rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }
    
    .score-excellent {
        color: #10b981;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }
    .score-good {
        color: #6366f1;
        text-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
    }
    .score-average {
        color: #f59e0b;
        text-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
    }
    .score-low {
        color: #ef4444;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }

    /* Tag Badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem 0.2rem;
    }
    .badge-match {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .badge-missing {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .badge-bonus {
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.35);
    }
    .badge-keyword {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }

    /* Card Panels */
    .panel-box {
        border-radius: 14px;
        padding: 1.25rem;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
    }
    
    /* Quick prompt button container */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0

# Document RAG state
if "doc_messages" not in st.session_state:
    st.session_state.doc_messages = []
if "doc_chat_history" not in st.session_state:
    st.session_state.doc_chat_history = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# CareerGuide AI state
if "career_analysis" not in st.session_state:
    st.session_state.career_analysis = None
if "career_messages" not in st.session_state:
    st.session_state.career_messages = []
if "career_chat_history" not in st.session_state:
    st.session_state.career_chat_history = []
if "active_resume_text" not in st.session_state:
    st.session_state.active_resume_text = ""
if "active_jd_text" not in st.session_state:
    st.session_state.active_jd_text = ""
if "uploaded_resume_text" not in st.session_state:
    st.session_state.uploaded_resume_text = ""
if "uploaded_jd_text" not in st.session_state:
    st.session_state.uploaded_jd_text = ""
if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False

# ---------------------------------------------------------
# Sidebar: Navigation & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **Navigation Mode**")
    app_mode = st.radio(
        "Choose Module:",
        ["🎯 CareerGuide AI (Resume & JD)", "📚 Document RAG Assistant"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("### 🦙 **Local Ollama Configuration**")
    ollama_model = st.selectbox(
        "Select Local Model",
        ["llama3.2:1b", "llama3:8b"],
        index=0,
        help="Choose a model already downloaded in Ollama."
    )
    embedding_model = "nomic-embed-text:latest"
    st.caption("🟢 Uses your local Ollama server. No Gemini API key required.")

    st.divider()
    st.markdown("### 📊 **Usage Tracker**")
    col1, col2 = st.columns(2)
    col1.metric("Input Tokens (not tracked)", f"{st.session_state.total_input_tokens:,}")
    col2.metric("Output Tokens (not tracked)", f"{st.session_state.total_output_tokens:,}")
    st.metric("Total Tokens Used", f"{st.session_state.total_input_tokens + st.session_state.total_output_tokens:,}")
    
    if st.button("🔄 Reset Token Stats", use_container_width=True):
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.rerun()

# Ollama local model configuration
client = True  # Ollama is used locally; API key is not required.

# ---------------------------------------------------------
# Helper Functions: Extraction & Embeddings
# ---------------------------------------------------------
@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

chroma_client = get_chroma_client()

def extract_text_from_file(uploaded_file):
    """Extract clean text from PDF, DOCX, or TXT file."""
    if uploaded_file is None:
        return ""
    
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip()
        
    elif filename.endswith(".docx"):
        if not DOCX_AVAILABLE:
            st.error("python-docx is not available. Please install it to parse DOCX files.")
            return ""
        doc = docx.Document(uploaded_file)
        full_text = [para.text for para in doc.paragraphs if para.text]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text:
                        full_text.append(cell.text)
        return "\n".join(full_text).strip()
        
    elif filename.endswith(".txt") or filename.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()
        
    return ""

def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def embed_text_content(text_content):
    """Generate a local embedding using Ollama."""
    result = ollama.embed(model=embedding_model, input=text_content)
    return result["embeddings"][0]

def generate_local_response(prompt, model=None, system=None, history=None, json_mode=False):
    """Generate a response from a local Ollama model."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    if prompt:
        messages.append({"role": "user", "content": prompt})
    kwargs = {"model": model or ollama_model, "messages": messages}
    if json_mode:
        kwargs["format"] = "json"
        kwargs["options"] = {"temperature": 0.2}
    response = ollama.chat(**kwargs)
    return response["message"]["content"]

# Sample Resume and Job Description for 1-click testing
SAMPLE_RESUME = """Alex Rivera
Senior Software Engineer & AI Practitioner
Email: alex.rivera@example.com | GitHub: github.com/alexrivera | LinkedIn: linkedin.com/in/alexrivera

PROFESSIONAL SUMMARY
Dynamic Full-Stack and Machine Learning Engineer with 5+ years of experience designing, developing, and deploying scalable distributed web applications and intelligent RAG systems. Proficient in Python, TypeScript, React, FastApi, Streamlit, and vector databases (ChromaDB, Pinecone). Led microservice migrations resulting in 40% latency reduction.

TECHNICAL SKILLS
- Languages: Python, JavaScript/TypeScript, SQL, HTML/CSS
- Frameworks & Libs: FastAPI, Flask, React, Next.js, Node.js, LangChain, PyTorch, Pandas, Scikit-learn
- AI & LLMs: Gemini API, OpenAI GPT-4, Retrieval-Augmented Generation (RAG), Vector Embeddings, ChromaDB, Prompt Engineering
- Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, Kubernetes, CI/CD, Git, PostgreSQL, Redis

PROFESSIONAL EXPERIENCE
Senior Software Engineer | TechPulse Solutions (2022 - Present)
- Architected and deployed an AI-driven enterprise knowledge search system using Python, ChromaDB, and Google Gemini API, indexing over 500,000 corporate documents and reducing search turnaround time by 65%.
- Built high-performance microservices using FastAPI and Docker, handling 15M+ monthly API requests with 99.98% uptime.
- Mentored a team of 4 junior developers in unit testing, code reviews, and containerization best practices.

Software Engineer | CloudInnovate Labs (2019 - 2022)
- Built interactive customer dashboards in React and Python, boosting user retention by 28%.
- Integrated RESTful APIs with relational databases (PostgreSQL) and optimized complex SQL queries to reduce query times by 35%.
- Implemented automated CI/CD deployment pipelines using GitHub Actions.

EDUCATION
Bachelor of Science in Computer Science | State University (2015 - 2019)
- Graduated Magna Cum Laude (GPA: 3.8/4.0)
- Certifications: AWS Certified Solutions Architect - Associate
"""

SAMPLE_JD = """Job Title: Senior AI / Full-Stack Engineer
Company: Apex Innovations
Location: Remote / Hybrid

About The Role:
We are seeking an experienced Senior AI / Full-Stack Engineer to lead the development of our next-generation generative AI products. In this role, you will design robust RAG architectures, build scalable backends with FastAPI, integrate modern web frontends, and collaborate with product teams to build intelligent AI-assisted workflows.

Key Responsibilities:
- Design and deploy production-grade RAG (Retrieval-Augmented Generation) pipelines and intelligent LLM agents.
- Develop scalable backend APIs using Python (FastAPI/Flask) and event-driven architectures.
- Work with vector databases (ChromaDB, Pinecone, or Milvus) and embedding models for semantic retrieval.
- Build clean, intuitive web interfaces using modern frameworks (React or Streamlit).
- Deploy containerized applications on cloud infrastructure (AWS or GCP) using Docker and Kubernetes.
- Drive code quality, security reviews, and mentor team members in best practices.

Required Qualifications:
- 4+ years of professional software development experience in Python and modern web technologies.
- Hands-on experience developing LLM-powered applications, prompt engineering, and RAG systems.
- Strong proficiency with Vector DBs (ChromaDB, Pinecone, Qdrant) and relational databases (PostgreSQL).
- Proficiency with Docker, Kubernetes, and Cloud platforms (AWS or GCP).
- Excellent communication skills, agile mindset, and passion for artificial intelligence.

Nice to Have:
- Experience with LangChain, LlamaIndex, or Google GenAI SDK.
- Knowledge of MLOps, CI/CD pipelines, and automated testing frameworks.
- Familiarity with TypeScript and GraphQL.
"""

# =========================================================
# MODULE 1: 🎯 CAREERGUIDE AI
# =========================================================
if app_mode == "🎯 CareerGuide AI (Resume & JD)":
    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🎯 CareerGuide AI</h1>
        <p class="hero-subtitle">Smart Resume & Job Description Compatibility Matcher, Gap Analyzer, and Interactive Career Coach</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not client:
        st.warning("⚠️ Start Ollama and download the selected model to use CareerGuide AI.")
    
    # Input Stage
    st.markdown("### 📋 Step 1: Provide Resume & Job Description")
    
    # 1-click sample button
    sample_col1, sample_col2 = st.columns([3, 1])
    with sample_col2:
        if st.button("✨ Load Sample Resume & JD", help="Quickly test with sample senior AI engineer data"):
            st.session_state.active_resume_text = SAMPLE_RESUME
            st.session_state.active_jd_text = SAMPLE_JD
            st.session_state.sample_loaded = True
            st.rerun()
            
    col_resume, col_jd = st.columns(2)
    
    with col_resume:
        st.markdown("#### 📄 **Candidate Resume**")
        resume_tab_upload, resume_tab_paste = st.tabs(["📁 Upload File (PDF/DOCX/TXT)", "✍️ Paste Text"])
        
        uploaded_resume_content = ""
        with resume_tab_upload:
            uploaded_resume = st.file_uploader(
                "Upload Resume", 
                type=["pdf", "docx", "txt"], 
                key="resume_uploader",
                help="PDF, Word Document (.docx), or plain text"
            )
            if uploaded_resume:
                uploaded_resume_content = extract_text_from_file(uploaded_resume)
                if uploaded_resume_content:
                    st.session_state.uploaded_resume_text = uploaded_resume_content
                    st.success(f"Loaded {uploaded_resume.name} ({len(uploaded_resume_content):,} characters)")
                    with st.expander("👁️ Preview Extracted Resume Text", expanded=False):
                        st.caption(uploaded_resume_content[:800] + ("..." if len(uploaded_resume_content) > 800 else ""))
            else:
                st.session_state.uploaded_resume_text = ""
                    
        with resume_tab_paste:
            pasted_resume = st.text_area(
                "Resume Content",
                value=st.session_state.active_resume_text if not uploaded_resume else "",
                height=220,
                placeholder="Paste the full text of your resume here...",
                key="resume_pasted_area"
            )

        # Resolve active resume text with proper precedence
        if uploaded_resume_content.strip():
            st.session_state.active_resume_text = uploaded_resume_content.strip()
        elif pasted_resume.strip():
            st.session_state.active_resume_text = pasted_resume.strip()
        elif not uploaded_resume and not pasted_resume.strip() and not st.session_state.get("sample_loaded", False):
            st.session_state.active_resume_text = ""

    with col_jd:
        st.markdown("#### 💼 **Target Job Description (JD)**")
        jd_tab_upload, jd_tab_paste = st.tabs(["📁 Upload File (PDF/TXT)", "✍️ Paste JD Text"])
        
        uploaded_jd_content = ""
        with jd_tab_upload:
            uploaded_jd = st.file_uploader(
                "Upload JD Document", 
                type=["pdf", "docx", "txt"], 
                key="jd_uploader",
                help="PDF, Word Document (.docx), or plain text"
            )
            if uploaded_jd:
                uploaded_jd_content = extract_text_from_file(uploaded_jd)
                if uploaded_jd_content:
                    st.session_state.uploaded_jd_text = uploaded_jd_content
                    st.success(f"Loaded {uploaded_jd.name} ({len(uploaded_jd_content):,} characters)")
                    with st.expander("👁️ Preview Extracted JD Text", expanded=False):
                        st.caption(uploaded_jd_content[:800] + ("..." if len(uploaded_jd_content) > 800 else ""))
            else:
                st.session_state.uploaded_jd_text = ""

        with jd_tab_paste:
            pasted_jd = st.text_area(
                "Job Description Content",
                value=st.session_state.active_jd_text if not uploaded_jd else "",
                height=220,
                placeholder="Paste the target job description, requirements, and responsibilities here...",
                key="jd_pasted_area"
            )

        # Resolve active JD text with proper precedence
        if uploaded_jd_content.strip():
            st.session_state.active_jd_text = uploaded_jd_content.strip()
        elif pasted_jd.strip():
            st.session_state.active_jd_text = pasted_jd.strip()
        elif not uploaded_jd and not pasted_jd.strip() and not st.session_state.get("sample_loaded", False):
            st.session_state.active_jd_text = ""

    # Analysis Action Button & Status Indicator
    st.markdown("<br>", unsafe_allow_html=True)
    resume_valid = bool(st.session_state.active_resume_text and st.session_state.active_resume_text.strip())
    jd_valid = bool(st.session_state.active_jd_text and st.session_state.active_jd_text.strip())
    api_valid = bool(client)

    analyze_btn_disabled = not (resume_valid and jd_valid and api_valid)
    
    if analyze_btn_disabled:
        missing_items = []
        if not resume_valid:
            missing_items.append("Candidate Resume")
        if not jd_valid:
            missing_items.append("Target Job Description")
        if not api_valid:
            missing_items.append("Local Ollama model")
        st.warning(f"⚠️ To unlock analysis, please provide: **{', '.join(missing_items)}**")
    else:
        st.caption(f"🟢 **Ready to analyze**: Resume ({len(st.session_state.active_resume_text):,} chars) & Job Description ({len(st.session_state.active_jd_text):,} chars)")
    
    if st.button("🚀 Run Compatibility Matching & Deep Gap Analysis", type="primary", use_container_width=True, disabled=analyze_btn_disabled):
        with st.spinner("Analyzing resume against job requirements with local Ollama..."):
            prompt = f"""
You are an expert Executive Recruiter, ATS Algorithm Specialist, and Senior Career Strategist.
Analyze the following Candidate Resume against the Target Job Description with high precision.

CANDIDATE RESUME:
\"\"\"
{st.session_state.active_resume_text}
\"\"\"

TARGET JOB DESCRIPTION:
\"\"\"
{st.session_state.active_jd_text}
\"\"\"

Provide your thorough analysis strictly in valid JSON format matching this exact schema:
{{
  "overall_score": 85,
  "match_level": "Strong Match",
  "category_scores": {{
    "technical_skills": 88,
    "experience_relevance": 82,
    "education_certifications": 90,
    "role_alignment": 85,
    "ats_keyword_density": 80
  }},
  "executive_summary": "A concise 2-3 sentence assessment of the candidate's fit.",
  "matched_skills": ["Skill 1", "Skill 2", "Skill 3"],
  "missing_skills": ["Missing Skill 1", "Missing Skill 2"],
  "bonus_skills": ["Bonus Skill 1", "Bonus Skill 2"],
  "key_strengths": [
    "Strength 1 with details",
    "Strength 2 with details",
    "Strength 3 with details"
  ],
  "critical_gaps": [
    "Gap or weakness 1",
    "Gap or weakness 2"
  ],
  "ats_recommendations": [
    "Actionable tip 1 to optimize resume keywords and bullet points",
    "Actionable tip 2 to improve ATS score"
  ],
  "tailored_interview_prep": [
    {{
      "question": "Behavioral or technical question hiring manager will ask",
      "focus": "Why they ask this (testing your gap or key skill)",
      "recommended_framing": "How the candidate should structure the answer using their experience"
    }}
  ]
}}

Return ONLY the raw JSON object. Do not include markdown ticks or additional commentary.
"""
            try:
                raw_text = generate_local_response(prompt, json_mode=True).strip()
                # Clean any markdown code blocks if present
                clean_json = re.sub(r"^```(?:json)?\s*", "", raw_text)
                clean_json = re.sub(r"\s*```$", "", clean_json)
                
                analysis_data = json.loads(clean_json)
                st.session_state.career_analysis = analysis_data
                
                # Reset career chatbot context to this new report
                st.session_state.career_messages = [
                    {
                        "role": "assistant",
                        "content": f"👋 Hello! I have completed the compatibility analysis between your resume and the target job description. Your overall match score is **{analysis_data.get('overall_score', 'N/A')}%** ({analysis_data.get('match_level', '')}). Ask me anything about tailoring your resume, explaining skill gaps, or preparing for interviews!"
                    }
                ]
                st.session_state.career_chat_history = []
                st.success("✅ Analysis Complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to generate analysis: {e}")

    # Display Analysis Results
    if st.session_state.career_analysis:
        analysis = st.session_state.career_analysis
        score = analysis.get("overall_score", 0)
        match_level = analysis.get("match_level", "Match Score")
        
        score_class = "score-excellent" if score >= 80 else ("score-good" if score >= 65 else ("score-average" if score >= 50 else "score-low"))
        
        st.divider()
        st.markdown("### 📊 Compatibility Match Dashboard")
        
        dash_col1, dash_col2 = st.columns([1, 2])
        
        with dash_col1:
            st.markdown(f"""
            <div class="score-hero-box">
                <div style="text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.85rem; color: #94a3b8;">Compatibility Score</div>
                <div class="score-number {score_class}">{score}%</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">{match_level}</div>
                <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #cbd5e1;">Evaluated against job requirements & ATS standards</div>
            </div>
            """, unsafe_allow_html=True)
            
        with dash_col2:
            st.markdown("#### 🎯 Dimensional Breakdown")
            cats = analysis.get("category_scores", {})
            for label, key in [
                ("Technical Skills Match", "technical_skills"),
                ("Experience & Seniority Relevance", "experience_relevance"),
                ("Education & Certifications", "education_certifications"),
                ("Role Alignment & Responsibilities", "role_alignment"),
                ("ATS Keyword Density", "ats_keyword_density")
            ]:
                val = cats.get(key, 0)
                st.write(f"**{label}** — {val}%")
                st.progress(val / 100.0)

        # Executive Summary
        st.markdown(f"""
        <div class="panel-box" style="margin-top: 1rem; border-left: 4px solid #6366f1;">
            <strong style="color: #818cf8; font-size: 1rem;">📌 Executive Summary:</strong><br>
            <span style="font-size: 0.95rem; line-height: 1.6;">{analysis.get('executive_summary', 'No summary available.')}</span>
        </div>
        """, unsafe_allow_html=True)

        # Skills & Gaps Breakdown
        st.markdown("#### 🔍 Skills & Keyword Alignment")
        sk_col1, sk_col2, sk_col3 = st.columns(3)
        
        with sk_col1:
            st.markdown("**✅ Matched Skills**")
            matched = analysis.get("matched_skills", [])
            if matched:
                badges = "".join([f'<span class="badge badge-match">✓ {s}</span>' for s in matched])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.caption("No strong skill matches identified.")
                
        with sk_col2:
            st.markdown("**⚠️ Missing / Skill Gaps**")
            missing = analysis.get("missing_skills", [])
            if missing:
                badges = "".join([f'<span class="badge badge-missing">✗ {s}</span>' for s in missing])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.caption("No critical gaps detected! Great alignment.")
                
        with sk_col3:
            st.markdown("**💡 Bonus Value Add Skills**")
            bonus = analysis.get("bonus_skills", [])
            if bonus:
                badges = "".join([f'<span class="badge badge-bonus">+ {s}</span>' for s in bonus])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.caption("No specific bonus skills noted.")

        # Strengths & Recommendations Tabs
        st.markdown("<br>", unsafe_allow_html=True)
        tab_rec, tab_str, tab_interview = st.tabs(["🚀 ATS Optimization & Action Plan", "💪 Strengths & Gaps Analysis", "🎤 Tailored Interview Prep Kit"])
        
        with tab_rec:
            st.markdown("##### 📝 How to Boost Your Match Score & Beat ATS Filters")
            recs = analysis.get("ats_recommendations", [])
            for i, r in enumerate(recs):
                st.markdown(f"**{i+1}.** {r}")
                
        with tab_str:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🌟 Key Strengths Identified")
                for s in analysis.get("key_strengths", []):
                    st.markdown(f"- {s}")
            with c2:
                st.markdown("##### ⚠️ Critical Gaps to Address")
                for g in analysis.get("critical_gaps", []):
                    st.markdown(f"- {g}")

        with tab_interview:
            st.markdown("##### 🎯 High-Probability Interview Questions For This Specific Role")
            questions = analysis.get("tailored_interview_prep", [])
            for i, q in enumerate(questions):
                with st.expander(f"Q{i+1}: {q.get('question', 'Interview Question')}", expanded=(i==0)):
                    st.markdown(f"**Why the interviewer asks this:** *{q.get('focus', '')}*")
                    st.markdown(f"**Recommended Strategy (STAR Framework):**\n{q.get('recommended_framing', '')}")

        # -----------------------------------------------------
        # Dedicated Career Coach Chatbot Section
        # -----------------------------------------------------
        st.divider()
        st.markdown("### 💬 Interactive Career Coach & Resume Copilot")
        st.caption("Ask questions about tailoring your resume, explaining gaps, drafting cover letters, or role-playing interview scenarios.")
        
        # Quick Action Prompts
        st.markdown("**Quick Prompts:**")
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        quick_prompt = None
        if qcol1.button("✍️ Tailor My Experience Section", use_container_width=True):
            quick_prompt = "Rewrite the most relevant project or experience bullet points from my resume to specifically highlight the skills required in this Job Description using strong action verbs and quantified impact metrics."
        if qcol2.button("✉️ Draft Targeted Cover Letter", use_container_width=True):
            quick_prompt = "Write a compelling, professional 3-paragraph cover letter tailored to this job description that showcases my strengths and enthusiastically frames how I solve their key challenges."
        if qcol3.button("🛡️ How to Address Skill Gaps", use_container_width=True):
            quick_prompt = f"How should I address the missing skills ({', '.join(analysis.get('missing_skills', [])[:3])}) during an interview without looking underqualified?"
        if qcol4.button("🎤 60-Second Elevator Pitch", use_container_width=True):
            quick_prompt = "Create a punchy, memorable 60-second elevator pitch introducing myself for this specific role."

        # Display Career Chat messages
        for msg in st.session_state.career_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Handle chat input or quick prompt
        career_user_prompt = st.chat_input("Ask your Career Coach anything...") or quick_prompt
        
        if career_user_prompt and client:
            st.session_state.career_messages.append({"role": "user", "content": career_user_prompt})
            with st.chat_message("user"):
                st.markdown(career_user_prompt)
                
            coach_system_instruction = f"""
You are CareerGuide AI, a world-class executive career coach, technical recruiter, and resume strategist.
You are assisting a candidate who is targeting a specific job.

CONTEXT:
Candidate Resume:
{st.session_state.active_resume_text}

Target Job Description:
{st.session_state.active_jd_text}

Compatibility Score: {score}% ({match_level})
Matched Skills: {', '.join(analysis.get('matched_skills', []))}
Missing Skills: {', '.join(analysis.get('missing_skills', []))}

Provide direct, actionable, concrete advice. When drafting bullet points or cover letters, format them with clean Markdown so the user can easily copy and paste them.
"""
            st.session_state.career_chat_history.append(
                {"role": "user", "content": career_user_prompt}
            )
            
            with st.chat_message("assistant"):
                with st.spinner("Career Coach is writing..."):
                    try:
                        coach_answer = generate_local_response(
                            prompt="",
                            system=coach_system_instruction,
                            history=st.session_state.career_chat_history
                        )
                            
                        st.markdown(coach_answer)
                        
                        st.session_state.career_chat_history.append(
                            {"role": "assistant", "content": coach_answer}
                        )
                        st.session_state.career_messages.append({"role": "assistant", "content": coach_answer})
                        
                        if len(st.session_state.career_chat_history) > 20:
                            st.session_state.career_chat_history = st.session_state.career_chat_history[-20:]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Chat generation error: {e}")

# =========================================================
# MODULE 2: 📚 DOCUMENT RAG ASSISTANT (ORIGINAL FAQ BOT)
# =========================================================
else:
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">📚 Document RAG Assistant</h1>
        <p class="hero-subtitle">Upload any document, index embeddings with ChromaDB, and query with citation-backed AI answers</p>
    </div>
    """, unsafe_allow_html=True)
    
    rag_col1, rag_col2 = st.columns([1, 2])
    
    with rag_col1:
        st.markdown("#### 📄 **Upload & Index Document**")
        uploaded_doc = st.file_uploader("Upload PDF or Document", type=["pdf", "docx", "txt"], key="rag_file_uploader")
        
        if uploaded_doc and client:
            if st.session_state.current_file != uploaded_doc.name:
                with st.spinner("Processing & indexing document into ChromaDB..."):
                    extracted_text = extract_text_from_file(uploaded_doc)
                    if not extracted_text:
                        st.error("Could not extract text from document.")
                    else:
                        chunks = chunk_text(extracted_text, chunk_size=1500, overlap=200)
                        collection_name = f"doc_{abs(hash(uploaded_doc.name)) % 100000}"
                        
                        try:
                            chroma_client.delete_collection(collection_name)
                        except Exception:
                            pass
                            
                        collection = chroma_client.create_collection(name=collection_name)
                        
                        # Index chunks with progress bar
                        prog_bar = st.progress(0.0)
                        status_text = st.empty()
                        
                        for i, chk in enumerate(chunks):
                            emb = embed_text_content(chk)
                            collection.add(
                                documents=[chk],
                                embeddings=[emb],
                                ids=[f"chunk_{i}"]
                            )
                            prog_bar.progress((i + 1) / len(chunks))
                            status_text.caption(f"Indexed chunk {i+1} / {len(chunks)}")
                            
                        st.session_state.current_file = uploaded_doc.name
                        st.session_state.collection_name = collection_name
                        st.session_state.num_chunks = len(chunks)
                        st.session_state.doc_messages = []
                        st.session_state.doc_chat_history = []
                        status_text.empty()
                        prog_bar.empty()
                        st.success(f"✅ Indexed {len(chunks)} chunks from {uploaded_doc.name}")
                        
        if st.session_state.current_file:
            st.info(f"**Active Document:** {st.session_state.current_file}\n\n**Indexed Chunks:** {st.session_state.get('num_chunks', 0)}")
            if st.button("🗑️ Clear Active Document Index"):
                st.session_state.current_file = None
                st.session_state.collection_name = None
                st.session_state.doc_messages = []
                st.session_state.doc_chat_history = []
                st.rerun()
                
    with rag_col2:
        st.markdown("#### 💬 **Chat with Your Document**")
        
        if not uploaded_doc and not st.session_state.current_file:
            st.info("👆 Upload a document on the left to start asking questions.")
            
        # Display Document RAG Chat Messages
        for msg in st.session_state.doc_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                    with st.expander("🔍 Verified Document Citations", expanded=False):
                        for idx, src in enumerate(msg["sources"]):
                            st.markdown(f"**Source Chunk {idx+1}:**")
                            st.caption(src[:450] + "..." if len(src) > 450 else src)
                            if idx < len(msg["sources"]) - 1:
                                st.divider()
                                
        # Chat Input for Document RAG
        if st.session_state.collection_name and client:
            if doc_prompt := st.chat_input("Ask a question about the uploaded document..."):
                st.session_state.doc_messages.append({"role": "user", "content": doc_prompt})
                with st.chat_message("user"):
                    st.markdown(doc_prompt)
                    
                collection = chroma_client.get_collection(st.session_state.collection_name)
                query_emb = embed_text_content(doc_prompt)
                
                results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=min(3, st.session_state.get("num_chunks", 3))
                )
                
                retrieved_chunks = results['documents'][0] if results['documents'] else []
                context_str = "\n\n".join(retrieved_chunks)
                
                rag_system_instruction = f"""
You are a knowledgeable, factual assistant answering questions strictly based on the provided document context.
If the answer cannot be found in the context below, state clearly: "I couldn't find that information in the uploaded document."

DOCUMENT CONTEXT:
{context_str}
"""
                st.session_state.doc_chat_history.append(
                    {"role": "user", "content": doc_prompt}
                )
                
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving facts & generating answer..."):
                        try:
                            rag_answer = generate_local_response(
                                prompt="",
                                system=rag_system_instruction,
                                history=st.session_state.doc_chat_history
                            )
                                
                            st.markdown(rag_answer)
                            
                            # Expandable sources
                            if retrieved_chunks:
                                with st.expander("🔍 Verified Document Citations", expanded=False):
                                    for idx, src in enumerate(retrieved_chunks):
                                        st.markdown(f"**Source Chunk {idx+1}:**")
                                        st.caption(src[:450] + "..." if len(src) > 450 else src)
                                        if idx < len(retrieved_chunks) - 1:
                                            st.divider()
                                            
                            st.session_state.doc_chat_history.append(
                                {"role": "assistant", "content": rag_answer}
                            )
                            st.session_state.doc_messages.append({
                                "role": "assistant",
                                "content": rag_answer,
                                "sources": retrieved_chunks
                            })
                            
                            if len(st.session_state.doc_chat_history) > 20:
                                st.session_state.doc_chat_history = st.session_state.doc_chat_history[-20:]
                            st.rerun()
                        except Exception as e:
                            st.error(f"RAG generation error: {e}")