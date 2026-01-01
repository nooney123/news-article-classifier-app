from flask import Flask, render_template, request
import torch
from model_utils import load_model, predict_headline

app = Flask(__name__)

# Load model & tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER, MODEL = load_model(device=DEVICE)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        headline = request.form["headline"]
        preds = predict_headline(headline, TOKENIZER, MODEL, device=DEVICE, topk=3)
        prediction = {"headline": headline, "preds": preds}
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
