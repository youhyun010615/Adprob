import csv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CSV_PATH = "input.csv"

HEADERS = [
    "text", "sponsored_keywords_count", "positive_bias_keywords", "external_links_count",
    "text_length_log", "most_common_word_freq_ratio", "price_mentions", "first_person_ratio",
    "lexical_diversity_ratio", "noun_ratio", "sentiment_score",
    "sponsored_position_ratio", "special_exclamation", "has_shopping_link", "image_density_log",
    "image_count", "is_ad"
]

def get_next_id():
    if not os.path.exists(CSV_PATH):
        return 1
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lines = list(csv.reader(f))
        if len(lines) <= 1:
            return 1
        try:
            last_id = int(lines[-1][0])
            return last_id + 1
        except:
            return 1

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    blog_text = data.get("content", "")
    img_count = data.get("images", 0)
    is_ad = data.get("is_ad", -1)

    new_id = get_next_id()

    row = [
        blog_text,        
        "", "", "", "",   
        "", "", "", "",
        "", "", "", "",
        "", "",
        img_count,        
        is_ad             
    ]

    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADERS)
        writer.writerow(row)

    return jsonify({"result": f"✅ 저장 완료 - ID: {new_id}"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
