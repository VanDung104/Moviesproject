# 🎬 Movie AI Chatbot

A hybrid movie assistant application featuring a futuristic Neon UI. This project combines modern **Generative AI (RAG)** with traditional **Machine Learning (TF-IDF)** to provide accurate movie recommendations.

## 🚀 Key Features

### 1. 🔍 Semantic Search (RAG)
Allows users to search for movies using natural language descriptions (e.g., *"movies about time travel paradox"*).
* **Workflow:** User Query (Vietnamese) → Translation (Helsinki-NLP) → Embedding (Sentence-Transformers) → Vector Search (Pinecone) → Answer Generation (Google Gemini).
* **Technology:** LangChain, Pinecone, Google Generative AI.

### 2. ⭐ Similarity Recommendation (TF-IDF)
Fast, content-based recommendation engine that suggests movies similar to a given title based on genres and keywords.
* **Workflow:** Mathematical calculation of cosine similarity between movie vectors.
* **Technology:** Scikit-learn, Pandas, Dill (No LLM required).

### 3. 🎨 Interactive UI
* **Design:** Cyberpunk/Neon aesthetic using Tailwind CSS.
* **UX:** Real-time typing effects, Markdown rendering (Marked.js), and responsive layout.

---

## 🛠 Tech Stack

* **Frontend:** HTML5, JavaScript, Tailwind CSS.
* **Backend:** Python, Flask.
* **AI & ML:**
    * **LLM:** Google Gemini 2.5 Flash.
    * **Vector Database:** Pinecone.
    * **Embeddings:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
    * **Classic ML:** TF-IDF Vectorizer, Cosine Similarity.

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/VanDung104/Moviesproject.git
   cd movie-ai-chatbot
