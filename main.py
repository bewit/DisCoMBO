MAX_NUM_PROCESSES = 8

import os
import torch

os.environ["OMP_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
os.environ["MKL_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{MAX_NUM_PROCESSES}"
torch.set_num_threads(MAX_NUM_PROCESSES)
torch.set_num_interop_threads(MAX_NUM_PROCESSES)


from discombo.experiments import HPOExperiment, NASBench101Experiment, NASBench201Experiment,\
      JAHSBenchExperiment, NASBench101InteractiveExperiment, NASBench201InteractiveExperiment,\
      JAHSBenchInteractiveExperiment, ToyExperiment, HPOBInteractiveExperiment, HPOBExperiment, QuadraticExperiment,\
      FCNetExperiment, LCExperiment, LCInteractiveExperiment, FCNetInteractiveExperiment, FlorisExperiment, \
    PVExperiment, PVInteractiveExperiment, FlorisInteractiveExperiment
import argparse
import os
from datetime import datetime
from discombo.utils import SEEDS, create_dir
from rtpt import RTPT
import numpy as np
import sklearn
from multiprocessing import Process

def get_experiment(args, seed):

    if args.exp == 'hpo':
        experiment = HPOExperiment(args.optimizer, args.task, seed=seed)
    elif args.exp == 'nas101':
        experiment = NASBench101Experiment(args.optimizer, args.task, args.handle_invalid_configs, seed=seed)
    elif args.exp == 'nas201':
        experiment = NASBench201Experiment(args.optimizer, args.task, seed=seed)
    elif args.exp == 'jahs':
        experiment = JAHSBenchExperiment(args.optimizer, args.task, seed=seed)
    elif args.exp == 'nas101_int':
        experiment = NASBench101InteractiveExperiment(args.optimizer, args.interaction_idx, args.task, seed=seed)
    elif args.exp == 'nas201_int':
        experiment = NASBench201InteractiveExperiment(args.optimizer, args.interaction_idx, args.task, seed=seed)
    elif args.exp == 'jahs_int':
        experiment = JAHSBenchInteractiveExperiment(args.optimizer, args.interaction_idx, args.task, seed=seed,
                                                    interventions_path=args.interventions_path)
    elif args.exp == 'toy':
        experiment = ToyExperiment(args.optimizer, function=args.task, rb_dims=3, seed=seed)
    elif args.exp == 'quadratic':
        experiment = QuadraticExperiment(args.optimizer, 3, seed=seed)
    elif args.exp == 'hpob':
        search_space_id, dataset_id = args.task.split(':')
        experiment = HPOBExperiment(args.optimizer, search_space_id, dataset_id, seed=seed)
    elif args.exp == 'hpob_int':
        search_space_id, dataset_id = args.task.split(':')
        experiment = HPOBInteractiveExperiment(args.optimizer, search_space_id, dataset_id, args.interaction_idx, seed=seed)
    elif args.exp == 'fcnet':
        experiment = FCNetExperiment(args.optimizer, args.task, seed=seed)
    elif args.exp == 'lc':
        experiment = LCExperiment(args.optimizer, args.task, seed=seed)
    elif args.exp == 'fcnet_int':
        experiment = FCNetInteractiveExperiment(args.optimizer, args.task, args.interaction_idx, seed=seed)
    elif args.exp == 'lc_int':
        experiment = LCInteractiveExperiment(args.optimizer, args.task, args.interaction_idx, seed=seed)
    elif args.exp == 'floris':
        experiment = FlorisExperiment(args.optimizer, seed=seed)
    elif args.exp == 'floris_int':
        experiment = FlorisInteractiveExperiment(args.optimizer, interaction_idx=args.interaction_idx, seed=seed)
    elif args.exp == 'pv':
        experiment = PVExperiment(args.optimizer, args.task, seed=seed)
    elif args.exp == 'pv_int':
        experiment = PVInteractiveExperiment(args.optimizer, args.task, args.interaction_idx, seed=seed)
    else:
        raise ValueError(f'No such experiment: {args.exp}.')
    return experiment


def run_experiment(args, seed_idx):
    exp_counter = 0
    rt = RTPT('JS', 'DisCoMBO', 1)
    rt.start()
    # set random seed for each experiment
    #seed = np.random.choice(np.array(SEEDS))
    seed = SEEDS[seed_idx]
    np.random.seed(seed)
    sklearn.random.seed(seed)
    
    exp_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    file_name = f'{args.exp}_{args.task}_{exp_time}_{seed}.csv'
    log_file = os.path.join(args.log_dir, file_name)
    experiment = get_experiment(args, seed)

    try:
        # PC fails sometimes, ignore these runs
        experiment.run()
        experiment.save(log_file)
        exp_counter += 1
        rt.step()
    except Exception as e:
        raise e
        print(f"Experiment failed with: {e}")

parser = argparse.ArgumentParser()

parser.add_argument('--num-experiments', default=500, type=int)
parser.add_argument('--optimizer', default='pc', type=str)
parser.add_argument('--exp', default='hpo', type=str)
parser.add_argument('--task', default='cifar10')
parser.add_argument('--log-dir', default='./', type=str)
parser.add_argument('--num-procs', default=5, type=int)
parser.add_argument('--interaction-idx', nargs='+', type=int, default=[-1])
parser.add_argument('--seed-offset', type=int, default=0)
parser.add_argument('--handle-invalid-configs', action='store_true', help='Provides fall back value for invalid configurations. Only applies to some experiments.')
parser.add_argument('--sampling-method', default='self_consistent', type=str, choices=['self_consistent', 'anneal'],
                    help='Sampling method for PC optimizer. anneal blends rejection and self-consistent sampling.')
parser.add_argument('--anneal-horizon', default=None, type=int,
                    help='Anneal horizon in iterations. Defaults to total iterations.')
parser.add_argument('--parallelize', default=False, type=bool)
parser.add_argument('--interventions-path', default='./interventions/jahs_cifar10.json', type=str,
                    help='Path to the interventions JSON file (currently only used by jahs_int).')

args = parser.parse_args()

#setup_environment()
create_dir(args.log_dir)

if not args.parallelize:
    print(f"Running {args.optimizer} in single thread mode.")
    exp_counter = 0
    while exp_counter < args.num_experiments:
        run_experiment(args, args.seed_offset + exp_counter)
        exp_counter += 1
else:
    num_procs = min(args.num_procs, args.num_experiments) # min ensures thtat there is at least one batch of processes
    num_batches = int(round(args.num_experiments / num_procs))
    print(f"Parallelizing {args.optimizer} with {num_procs} processes.")

    for b in range(num_batches):
        print(f"Start batch {b+1}/{num_batches} of experiments")
        seed_offset = b*num_procs
        processes = [Process(target=run_experiment, args=(args, seed_offset+i)) for i in range(num_procs)]
        for p in processes:
            p.start()

        for p in processes:
            p.join()
