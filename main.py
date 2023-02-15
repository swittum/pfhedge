import seaborn
from InputReader import InputReader
if __name__ == "__main__":
    seaborn.set_style("whitegrid")
    reader = InputReader("config5.yaml")
    handler = reader.load_config()
    handler.full_process()
