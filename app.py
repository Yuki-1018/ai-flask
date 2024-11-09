from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# ローカルの「model」フォルダからモデルとトークナイザーを読み込み
MODEL_PATH = "./model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model = model.to("cpu")  # CPUで動作させる

# 応答を生成する関数
def generate_response(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():  # 勾配計算を無効にしてメモリ節約
        outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_p=0.9, top_k=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# チャットエンドポイント (JSON API)
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = generate_response(user_input)
    return jsonify({"response": response})

# HTMLのレンダリング
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
