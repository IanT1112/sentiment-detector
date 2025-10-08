from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Modelo de análisis de sentimientos
sentiment_analyzer = pipeline("sentiment-analysis")

# Lista en memoria para guardar los comentarios
history = []

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        user_text = request.form['text']
        if user_text.strip():
            result = sentiment_analyzer(user_text)[0]
            history.insert(0, {
                "text": user_text,
                "label": result['label'],
                "score": result['score']
            })
    return render_template('index.html', result=result, history=history)

if __name__ == '__main__':
    app.run(debug=True)
