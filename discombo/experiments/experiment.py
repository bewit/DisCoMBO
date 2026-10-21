import pandas as pd
from ..benchmarks import BenchQueryResult, Benchmark
from ..optimizers import Optimizer

class Experiment:

    def __init__(self) -> None:
        pass

    def run(self):
        raise NotImplementedError('Experiment has no implementation for run')
    
class BenchmarkExperiment(Experiment):

    def __init__(self, benchmark: Benchmark, optimizer: Optimizer, seed=0) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.optimizer = optimizer
        self.seed = seed

    def save(self, file):
        print("============save experiment===================")
        print(file)
        run_history = self.optimizer.history
        res_cols = BenchQueryResult.SUPPORTED_METRICS
        prototype_cfg = run_history[0][0]
        cols = res_cols + list(prototype_cfg.keys())
        df_dict = {c: [] for c in cols}
        # add iteration column
        df_dict['iter'] = []
        for cfg, res, iteration in run_history:
            for k, v in cfg.items():
                df_dict[k].append(v)
            
            for k in res_cols:
                df_dict[k].append(res[k])
            
            df_dict['iter'].append(iteration)

        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(file)
