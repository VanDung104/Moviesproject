import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone
import dill
import difflib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Khởi động server, đang tải các model...")

app = Flask(__name__)

# === LOAD FILE .env ===
app_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(app_dir)
dotenv_path = os.path.join(parent_dir, '.env')

print(f"Đang tải file .env từ: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError(f"Lỗi: Không tìm thấy PINECONE_API_KEY tại {dotenv_path}")
else:
    print("✔️ Đã tải PINECONE_API_KEY thành công.")

# === KẾT NỐI PINECONE ===
print("Đang kết nối tới Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("movies")

# === LOAD MODEL EMBEDDING & TRANSLATION ===
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"

print(f"Đang tải model embedding: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print(f"Đang tải model dịch thuật: {TRANSLATION_MODEL}...")
translator = pipeline("translation", model=TRANSLATION_MODEL)

print("✅ Server sẵn sàng! Tất cả model đã được tải.")
print("=" * 50)


# === API: SEMANTIC SEARCH ===
@app.route('/search', methods=['GET'])
def search_movies():
    vietnamese_query = request.args.get('q')
    if not vietnamese_query:
        return jsonify({"error": "Vui lòng cung cấp query (ví dụ: /search?q=phim gì đó)"}), 400

    print(f"Nhận request: '{vietnamese_query}'")

    try:
        translated_result = translator(vietnamese_query)
        english_query = translated_result[0]['translation_text']
        print(f"   -> Đã dịch sang (EN): '{english_query}'")

        query_embedding = embedding_model.encode(english_query).tolist()
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        movie_results = []
        if results['matches']:
            for match in results['matches']:
                metadata = match.get('metadata', {})
                movie_results.append({
                    "id": match.get('id', 'N/A'),
                    "title": metadata.get('title', 'N/A'),
                    "genres": metadata.get('genres', 'Không rõ'),
                    "overview": metadata.get('overview', 'Không có mô tả'),
                    "score": match.get('score', 0.0)
                })

        return jsonify({
            "query_vietnamese": vietnamese_query,
            "query_english": english_query,
            "results": movie_results
        })

    except Exception as e:
        print(f"Lỗi: {e}")
        return jsonify({"error": "Đã xảy ra lỗi trên server"}), 500


# === LOAD MÔ HÌNH RECOMMENDER (TF-IDF) ===
try:
    with open('movie_recommender_2.pkl', 'rb') as file:
        movie_recommender = dill.load(file)
    print(">>> Mô hình recommender đã được tải thành công!")
except FileNotFoundError:
    print("!!! LỖI: Không tìm thấy tệp 'movie_recommender_2.pkl'.")
    movie_recommender = None
except Exception as e:
    print(f"!!! LỖI khi tải mô hình: {e}")
    movie_recommender = None


# === API: TF-IDF RECOMMENDER ===
@app.route('/recommend', methods=['POST'])
def recommend_movies():
    if movie_recommender is None:
        return jsonify({"error": "Mô hình recommender chưa sẵn sàng hoặc tải thất bại."}), 500

    data = request.get_json()
    if not data or 'title' not in data:
        return jsonify({"error": "Thiếu dữ liệu JSON. Cần key 'title'."}), 400

    movie_title = data['title'].strip()
    print(f"Nhận yêu cầu gợi ý cho: {movie_title}")

    # === DỊCH SANG TIẾNG ANH ===
    try:
        translated = translator(movie_title)
        movie_title_en = translated[0]['translation_text']
        print(f"   -> Đã dịch sang (EN): '{movie_title_en}'")
    except Exception as e:
        movie_title_en = movie_title
        print(f"   -> Không dịch được, dùng nguyên văn: '{movie_title_en}'")

    try:
        # === FUZZY MATCH ===
        all_titles = list(movie_recommender.df['title'].values)
        match = difflib.get_close_matches(movie_title_en, all_titles, n=1, cutoff=0.6)

        if not match:
            return jsonify({"error": f"Không tìm thấy phim '{movie_title}' trong cơ sở dữ liệu."}), 404

        real_title = match[0]
        recommendations = movie_recommender.recommend(real_title)

        if hasattr(recommendations, "tolist"):
            recommendations = recommendations.tolist()

        print(f"→ Kết quả gợi ý: {recommendations}")
        return jsonify({
            "query": movie_title,
            "matched_title": real_title,
            "recommendations": recommendations
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Lỗi khi gợi ý: {str(e)}"}), 500


# === SERVE FRONTEND ===
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
