import seaborn
from InputReader import InputReader
if __name__ == "__main__":
    print('Entering')
    seaborn.set_style("whitegrid")
    reader = InputReader("config.yaml")
    handler = reader.load_config()
    # backup_path = '/home/swittum/pfhedge/Backup_Parameters/params.pt'
    handler.full_process(backup=True)
    print(handler.hedger)
    handler.stock_diagrams(5)
