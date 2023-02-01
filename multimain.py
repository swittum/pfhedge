import seaborn
from InputReader import InputReader
if __name__ == "__main__":
    seaborn.set_style("whitegrid")
    reader = InputReader("multiconfig.yaml")
    handler = reader.load_multi_config()
    handler.fit()
    bench = handler.benchmark()
    handler.profit(bench)