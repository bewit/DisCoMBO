from . import *

class OptimizerFactory:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OptimizerFactory, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_optimizer(cls, optimizer_name, optimizer_cfg):
        if optimizer_name == 'pc':
            return PCOptimizer(**optimizer_cfg)
        elif optimizer_name == 'ls':
            return LocalSearchOptimizer(**optimizer_cfg)
        elif optimizer_name == 'rs':
            return RandomSearchOptimizer(**optimizer_cfg)
        elif optimizer_name == 'gp':
            return GPOptimizer(**optimizer_cfg)
        elif optimizer_name == 'smac':
            return SMACOptimizer(**optimizer_cfg)
        elif optimizer_name == 'pibo':
            return PiBOptimizer(**optimizer_cfg)
        elif optimizer_name == 'hyperband':
            return HyperbandOptimizer(**optimizer_cfg)
        elif optimizer_name == 'bopro':
            return BOPrOOptimizer(**optimizer_cfg)
        elif optimizer_name == 'optunabo':
            return OptunaBOOptimizer(**optimizer_cfg)
        elif optimizer_name == 'skoptbo':
            return SkOptBOOptimizer(**optimizer_cfg)
        elif optimizer_name == 'dynabo':
            return DynaBOOptimizer(**optimizer_cfg)
        elif 'mc_stratified' in optimizer_name:
            return MCOptimizer(**optimizer_cfg)
        elif optimizer_name == 'cmaes':
            return CMAESOptimizer(**optimizer_cfg)
        elif optimizer_name == 'hebo':
            return HEBOOptimizer(**optimizer_cfg)
        else:
            raise ValueError(f'No such optimizer available: {optimizer_name}')