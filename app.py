from flask import Flask, request, redirect, flash, render_template
import seaborn
import matplotlib.pyplot as plt
from InputReader import InputReader
from plotting_library import make_training_diagram, make_pl_diagram,make_multi_profit,make_stock_diagram, figure_to_string
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
            file.save('upload.yaml')
            return redirect('/')
    return render_template('upload_dialog.html')
@app.route('/hedge')
def run_hedging():
    reader = InputReader("upload.yaml")
    if not reader.is_multi():
        handler = reader.load_config()
        training = handler.fit()
        profit = handler.profit()
        bench = handler.benchmark()
        result_model = handler.eval(profit)
        result_benchmark = handler.dict_eval(bench)
        training_figure = figure_to_string(make_training_diagram(training))
        profit_figure = figure_to_string(make_pl_diagram(profit))
        benchmark_figures = {}
        for key,value in bench.items():
            benchmark_figures[key] = figure_to_string(make_pl_diagram(value))
        return render_template("results.html", profit=result_model, bench=result_benchmark, train_fig = training_figure, profit_fig = profit_figure, bench_figs = benchmark_figures)
    handler = reader.load_multi_config()
    handler.fit()
    bench = handler.benchmark()
    profit = handler.profit()
    results_figure = figure_to_string(make_multi_profit(profit,handler.params,bench))
    return render_template("multiresults.html", total_fig = results_figure)
@app.route('/generate')
def stock_diagrams():
    if 'samples' in request.args:
        try:
            n_paths = int(request.args['samples'])
        except ValueError:
            flash("Invalid number")
            return redirect('/generate')
        reader = InputReader('upload.yaml')
        if reader.is_multi():
            multi = reader.load_multi_config()
            handler = multi.handlers[0]
        else:
            handler = reader.load_config()
        handler.derivative.simulate(n_paths)
        prices = handler.derivative.underlier.spot
        figures = []
        for i in range(n_paths):
            fig = make_stock_diagram(prices[i,...])
            figures.append(figure_to_string(fig))
            plt.close(fig)
        return render_template('show_diagrams.html', figures=figures)
    else:
        return render_template("diagram_form.html")
if __name__ == "__main__":
    app.run('0.0.0.0',debug=True)