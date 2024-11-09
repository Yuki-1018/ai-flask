from transformers import AutoModelForCausalLM, AutoTokenizer

# モデル名と保存先ディレクトリ
MODEL_NAME = "rinna/japanese-gpt2-xsmall"
LOCAL_DIR = "./model"  # カレントディレクトリの中に「model」フォルダを作成

# モデルとトークナイザーをローカルにダウンロード
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=LOCAL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=LOCAL_DIR)

print("モデルが「model」フォルダに保存されました:", LOCAL_DIR)
