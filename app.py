from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# モデルとトークナイザーのロード
MODEL_NAME = "rinna/japanese-gpt2-xsmall"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to("cpu")  # GPUがない場合はCPUで実行
except Exception as e:
    print(f"モデルまたはトークナイザーの読み込み中にエラーが発生しました: {e}")
    tokenizer, model = None, None  # エラー時にNoneを代入

# 応答を生成する関数
def generate_response(prompt, max_length=50):
    if tokenizer is None or model is None:
        return "エラー: モデルが正しく読み込まれていません。"
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():  # 勾配計算を無効にしてメモリ節約
            outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_p=0.9, top_k=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"応答生成中にエラーが発生しました: {e}")
        return "エラー: 応答生成中に問題が発生しました。"

# チャットエンドポイント (JSON API)
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "エラー: メッセージが空です。"}), 400  # 入力が空の場合
    response = generate_response(user_input)
    return jsonify({"response": response})

# HTMLのレンダリング
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
