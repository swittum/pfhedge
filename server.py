import os, sys
from flask import Flask, request, redirect, flash, render_template, url_for
import seaborn
from InputReader import InputReader
from plotting_library import make_training_diagram, make_pl_diagram, figure_to_string
seaborn.set_style("whitegrid")
app = Flask(__name__, static_url_path='', static_folder='/')
app.secret_key = 'test'
@app.route('/')
def index():
    return render_template("main.html")
@app.route('/upload', methods=['GET','POST'])
def upload_config():
    if request.method == 'POST':
        if 'config' not in request.files:
           flash('Invalid request')
           return redirect(request.url)
        file = request.files['config']
        if file.filename == '':
            flash('No file sent')
            return redirect(request.url)
        if file:
            file.save('config.yaml')
            return redirect('/')
    return render_template('upload_dialog.html')
@app.route('/hedge')
def run_hedging():
    reader = InputReader("config.yaml")
    handler = reader.load_config()
    training = handler.fit()
    profit = handler.profit()
    bench = handler.benchmark()
    result_model = handler.eval(profit)
    result_benchmark = handler.dict_eval(bench)
    training_figure = figure_to_string(make_training_diagram(training))
    profit_figure = figure_to_string(make_pl_diagram(profit))
    benchmark_figures = {}
    for key in bench.keys():
        benchmark_figures[key] = figure_to_string(make_pl_diagram(bench[key]))
    return render_template("results.html", profit=result_model, bench=result_benchmark, train_fig = training_figure, profit_fig = profit_figure, bench_figs = benchmark_figures)
app.run(debug=True)