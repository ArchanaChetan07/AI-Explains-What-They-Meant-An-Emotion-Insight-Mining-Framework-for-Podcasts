from flask import Flask, render_template
from routes import predict_route

app = Flask(__name__)
app.register_blueprint(predict_route)

@app.route('/')
def home():
    return render_template("index.html")
