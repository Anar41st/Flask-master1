import pickle

import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_lab1"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"}]

loaded_model_lin_reg = pickle.load(open('models/Sweets', 'rb'))
loaded_model_log_reg = pickle.load(open('models/apple', 'rb'))
loaded_model_knn = pickle.load(open('models/apple_knn', 'rb'))
loaded_model_tree = pickle.load(open('models/apple_tree', 'rb'))

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_lab1", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Линейная регрессия", menu=menu, accurasy='', class_model='')
    if request.method == 'POST':
        X_param = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        pred = loaded_model_lin_reg.predict(X_param)
        metkost=0.83
        return render_template('lab1.html', title="Линейная регрессия", menu=menu, accurasy=metkost,
                               class_model=pred[0][0])

@app.route('/api', methods=['get'])
def get_confee():
    request_data = request.get_json()
    X_param = np.array([[float(request_data['list1']),
                       float(request_data['list2']),
                       float(request_data['list3'])]])
    pred = loaded_model_lin_reg.predict(X_param)

    return jsonify(sort=pred[0])

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu, accurasy='', class_model='')
    if request.method == 'POST':
        X_param = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        pred = loaded_model_log_reg.predict(X_param)
        metkost =0.69
        return render_template('lab2.html', title="Логистическая регрессия", accurasy=metkost, menu=menu,
                               class_model=pred)


@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Метод ближайших соседей kNN", menu=menu, accurasy='', class_model='')
    if request.method == 'POST':
        X_param = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        pred = loaded_model_knn.predict(X_param)
        metkost =1
        return render_template('lab3.html', title="Метод ближайших соседей kNN", accurasy=metkost, menu=menu,
                               class_model=pred)

@app.route("/p_lab4", methods=['POST', 'GET'])
def f_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Дерево решений", menu=menu, accurasy='', class_model='')
    if request.method == 'POST':
        X_param = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        pred = loaded_model_tree.predict(X_param)
        metkost=0.879
        return render_template('lab4.html', title="Дерево решений", accurasy=metkost, menu=menu,
                               class_model=pred)


if __name__ == "__main__":
    app.run(debug=True)
