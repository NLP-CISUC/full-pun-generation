import os

from flask import Flask, jsonify, request, session
from flask_cors import CORS

from auth import authenticate
from generation_processor import (get_all_headlines,
                                  get_generated_for_headline,
                                  process_generation_files)

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Change for production!
CORS(app, supports_credentials=True)


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if authenticate(username, password):
        session["username"] = username
        return jsonify({"status": "success", "message": "Login feito com sucesso!"}), 200
    else:
        return jsonify({"status": "failure", "message": "Usuário ou senha errados."}), 401


@app.route('/get_headlines', methods=['GET'])
def list_headlines():
    headlines = get_all_headlines()
    return jsonify(headlines), 200


@app.route('/get_generated', methods=['GET'])
def get_generated():
    # if "username" not in session:
    #     return jsonify({"status": "failure", "message": "Por favor, faça o login."}), 401
    headline_id = request.args.get("id")
    if headline_id:
        data = get_generated_for_headline(headline_id)
        if data:
            return jsonify(data), 200
        return jsonify({"status": "failure", "message": "Notícia não encontrada."}), 404
    return jsonify({"status": "failure", "message": "Escolha uma notícia."}), 400


if __name__ == '__main__':
    # Pre-process generation files (select best outputs)
    process_generation_files()
    app.run(debug=True)
