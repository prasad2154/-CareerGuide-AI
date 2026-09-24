# ⚡ CareerGuide AI & Document RAG Platform

An enterprise-grade Streamlit AI platform combining **Document RAG (Retrieval-Augmented Generation)** with an intelligent **CareerGuide AI Suite** for Resume & Job Description Compatibility Scoring, Gap Analysis, and an interactive Career Coach Copilot.

---

## 🌟 Key Features

### 1. 🎯 CareerGuide AI (Resume & Job Description Platform)
- **Multi-Format Ingestion**: Upload Resume (PDF, DOCX, TXT) and target Job Description (PDF, DOCX, TXT) or paste text directly.
- **1-Click Sample Demo**: Preloaded with a sample Senior AI / Full-Stack Engineer profile and JD for instant testing.
- **Compatibility Score Engine**: Overall percentage matching score with qualitative status (*Excellent Match*, *Strong Match*, *Moderate Match*, etc.).
- **Dimensional Breakdown**:
  - Technical & Hard Skills Match %
  - Experience & Seniority Level Relevance %
  - Education & Certifications %
  - Role Alignment & Responsibilities %
  - ATS Keyword Density %
- **Skill & Keyword Analysis**:
  - Matched Skills (highlighted with success badges)
  - Missing Skills & Gaps (flagged with alert badges)
  - Bonus Skills (candidate assets beyond JD requirements)
- **Executive Assessment**: Summary profile, key candidate strengths, and potential red flags.
- **ATS Optimization & Action Plan**: Concrete, actionable recommendations to improve resume ranking and pass ATS screening.
- **Tailored Interview Prep Kit**: High-probability behavioral and technical interview questions based on the candidate's specific background and JD gaps, with STAR-method framing tips.
- **Interactive Career Coach Chatbot**: Context-aware AI coach that drafts tailored cover letters, rewrites project bullet points, prepares elevator pitches, and advises on explaining skill gaps.

---

### 2. 📚 Document RAG Assistant (General Document QA)
- **Document Chunking & Vectorization**: Automated text chunking with customizable overlap.
- **ChromaDB Vector Store**: Semantic vector storage with high-speed retrieval.
- **Gemini Embeddings**: High-dimensional semantic vectors (`gemini-embedding-001` / `text-embedding-004`).
- **Citation-Backed Answers**: Chatbot answers questions strictly using document context, with expandable source chunks and citations.
- **Token Usage Tracker**: Live tracking of input and output token consumption with 1-click reset.

---

### 3. 🎨 Modern & Polished UI
- Plus Jakarta Sans typography with gradient hero banners.
- Glassmorphic card design and responsive multi-column layouts.
- Dynamic color-coded score gauges, progress meters, and badge pills.
- Sidebar with model selector (`gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`) and direct API key configuration.

---

## 🚀 Quick Start Guide

### 1. Setup Environment
```bash
# Activate your virtual environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Create or edit your `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Alternatively, you can paste your Gemini API key directly into the sidebar in the UI)*

### 3. Launch the Application
```bash
streamlit run 12_app.py
# or
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/) with Custom CSS
- **LLM & Embeddings**: [Google GenAI SDK](https://github.com/google-gemini/generative-ai-python) (Gemini Flash & Gemini Embedding models)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Document Parsers**: PyPDF2, python-docx
"# -CareerGuide-AI"

## Demo Images 
 <img width="1865" height="847" alt="image" src="https://github.com/user-attachments/assets/da9cbca2-54bb-4685-bdae-2c899c6f579f" />
<img width="1916" height="847" alt="Screenshot 2026-09-24 164115" src="https://github.com/user-attachments/assets/df8fc239-88cc-41b4-9c6e-6405c7a5c3fc" />



