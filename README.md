\# 🎬 Movie AI Chatbot



A hybrid movie assistant application featuring a futuristic Neon UI. This project combines modern \*\*Generative AI (RAG)\*\* with traditional \*\*Machine Learning (TF-IDF)\*\* to provide accurate movie recommendations.



\## 🚀 Key Features



\### 1. 🔍 Semantic Search (RAG)

Allows users to search for movies using natural language descriptions (e.g., \*"movies about time travel paradox"\*).

\* \*\*Workflow:\*\* User Query (Vietnamese) → Translation (Helsinki-NLP) → Embedding (Sentence-Transformers) → Vector Search (Pinecone) → Answer Generation (Google Gemini).

\* \*\*Technology:\*\* LangChain, Pinecone, Google Generative AI.



\### 2. ⭐ Similarity Recommendation (TF-IDF)

Fast, content-based recommendation engine that suggests movies similar to a given title based on genres and keywords.

\* \*\*Workflow:\*\* Mathematical calculation of cosine similarity between movie vectors.

\* \*\*Technology:\*\* Scikit-learn, Pandas, Dill (No LLM required).



\### 3. 🎨 Interactive UI

\* \*\*Design:\*\* Cyberpunk/Neon aesthetic using Tailwind CSS.

\* \*\*UX:\*\* Real-time typing effects, Markdown rendering (Marked.js), and responsive layout.



---



\## 🛠 Tech Stack



\* \*\*Frontend:\*\* HTML5, JavaScript, Tailwind CSS.

\* \*\*Backend:\*\* Python, Flask.

\* \*\*AI \& ML:\*\*

&nbsp;   \* \*\*LLM:\*\* Google Gemini 2.5 Flash.

&nbsp;   \* \*\*Vector Database:\*\* Pinecone.

&nbsp;   \* \*\*Embeddings:\*\* `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.

&nbsp;   \* \*\*Classic ML:\*\* TF-IDF Vectorizer, Cosine Similarity.



---



\## ⚙️ Installation \& Setup



1\.  \*\*Clone the repository\*\*

&nbsp;   ```bash

&nbsp;   git clone https://github.com/VanDung104/Moviesproject.git

&nbsp;   cd movie-ai-chatbot

&nbsp;   ```

2\.  \*\*Environment Configuration\*\*

&nbsp;   Create a `.env` file in the root directory and add your API keys:

&nbsp;   ```env

&nbsp;   PINECONE\_API\_KEY=your\_pinecone\_key

&nbsp;   GOOGLE\_API\_KEY=your\_google\_key

&nbsp;   ```



3\.  \*\*Run the Application\*\*

&nbsp;   ```bash

&nbsp;   python ap3.py

&nbsp;   ```

&nbsp;   Access the chatbot at: `http://127.0.0.1:5001/`



---



\## 📂 Project Structure



```text

├── app.py                   # Main Flask backend application

├── index.html               # Frontend interface

├── movie\_recommender\_2.pkl  # Pre-trained TF-IDF model (Pickle file)

├── .env                     # Environment variables



