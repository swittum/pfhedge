import seaborn
from InputReader import InputReader
if __name__ == "__main__":
    seaborn.set_style("whitegrid")
    reader = InputReader("config.yaml")
    handler = reader.load_config()
    handler.full_process()
    # SWIT: Create additional stock_diagrams
    handler.stock_diagrams(5)
