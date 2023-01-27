import yaml
from config_utils import make_underlier, make_derivative, make_hedge, make_model, make_criterion
from pfhedge.nn import Hedger
from utils import prepare_features
from HedgeHandler import HedgeHandler
class InputReader:
    def __init__(self, filename):
        stream = open(filename, mode="r", encoding="utf-8")
        self.config = yaml.safe_load(stream)
        stream.close()
    def load_config(self) -> HedgeHandler:
        underlier = make_underlier(self.config.get('underlier',{}))
        derivative = make_derivative(self.config.get('derivative',{}), underlier)
        hedge = make_hedge(self.config.get('hedge',{}),underlier)
        model = make_model(self.config.get('model',{}),len(hedge),derivative)
        loss = make_criterion(self.config.get('loss',{}))
        features = self.config.get('features', [])
        if features == []:
            NTB = self.config.get('model',{}).get('NTB',False)
            features = prepare_features(derivative,NTB)
        hedger = Hedger(model,features,loss)
        fit_params = self.config.get('training',{})
        profit_params = self.config.get('profit',{})
        criterion = make_criterion(self.config.get('criterion',{}))
        benchmark_params = self.config.get('benchmark',{})
        return HedgeHandler(hedger,derivative,hedge,fit_params,profit_params,criterion,benchmark_params)
