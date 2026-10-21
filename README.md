# DisCoMBO: Steering Expert-in-the-Loop Black Box Optimization via Distributional Conformance
This repository contains the code of `Steering Expert-in-the-Loop Black Box Optimization via Distributional Conformance` (https://arxiv.org/pdf/2609.36472).

## Abstract
Sequential Model-Based Optimization (SMBO) traditionally relies on Bayesian or ensembling surrogates for uncertainty quantification. While historically treated as fully data-driven, SMBO increasingly integrates external domain expertise to accelerate discovery. To overcome the opaque guidance and diminished integration fidelity of standard acquisition re-weighting, Probabilistic Circuits (PCs) have emerged as a generative surrogate alternative, enabling direct knowledge injection via conditional sampling. However, these generative routines lack the formal exploration-exploitation semantics required for rigorous optimization. We introduce the \textbf{Dis}tributional \textbf{Co}nformance Score (DisCo), a novel metric that unifies the flexibility and efficiency of PCs with a formal uncertainty framework. DisCo provides a bounded, $[0, 1]$-normalized measure of model "surprise" that (1) recovers properties comparable to kernel-based uncertainty known from, e.g., Gaussian Processes, while maintaining linear-time inference, and (2) enables accurate assessment of conformance of external knowledge w.r.t. model evidence. We then present DisCoMBO, a framework leveraging these properties for robust, knowledge-aware optimization. We prove that DisCoMBO is a zero-regret algorithm and demonstrate its effectiveness across diverse benchmarks from AutoML, material optimization, and wind park optimization.

## Setup
We recommend to run the code in a `Docker` container, we don't guarantee a working version without Docker.

To setup the code and run experiments do the following instructions:

1. Navigate to the root directory of the repository
2. Run `docker build --build-arg baseimage=python:3.9-slim --no-cache --tag discombo .`
3. Run `docker run -v [HOST_PATH]:/app/ -it discombo` to enter the Docker container.
4. Run `python main.py --exp [exp] --optimizer [optimizer]` to run the code (change options accordingly).

> Note: The build process can take 15-30 minutes, depending on your machine. Also, there will be quite a few **warnings** due to dependency incompatibilities which cannot be resolved currently. However, the codebase works as intended, only further extensions might be affected by dependency issues.

> Note: Once you start the container the first time, a setup script will be called. This loads all the benchmarks from various online sources and will download approximately 6GB of data. Depending on your internet connection, this can also take a while.

> Note: The JAHS benchmark requires relatively large RAM resources (>10GB). Thus, JAHS experiments might not be executable on machines with less memory.

For example, you can run the following command to run an experiment on the NAS-Bench-101 benchmark using a TPE BO optimizer from optuna:

`python main.py --exp nas101 --num-experiments 1 --optimizer optunabo --task cifar10 --log-dir ./data/nas101/`

This command will take approximately 10-20 minutes (depending on the machine) and produce a `.csv` logfile in the directory `/app/data/nas101/`.


Currently, the following experiments are implemented:

| Benchmark Name |   Option Name     | 5-Word Description                               | URL to Paper/GitHub                                                                                   |
|---------------|--------|--------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| NAS-Bench-101 |  nas101, nas101_int      | First public architecture dataset for NAS research. | [Paper](https://arxiv.org/abs/1902.09635), [GitHub](https://github.com/google-research/nasbench)     |
| NAS-Bench-201 |  nas201, nas201_int      | Extends NAS-Bench-101 with different search space. | [Paper](https://arxiv.org/abs/2001.00326), [GitHub](https://github.com/D-X-Y/NAS-Bench-201)          |
| JAHS-Bench-201|  jahs, jahs_int      | Joint architecture and hyperparameter search benchmark. | [Paper](https://arxiv.org/abs/2206.10555), [GitHub](https://github.com/automl/jahs_bench_201)        |
| LCBench       |   lc, lc_int     | Benchmark for low-cost hyperparameter optimization. | [Paper](https://arxiv.org/abs/2006.13799), [GitHub](https://github.com/automl/LCBench)               |
| FCNet         |   fcnet, fcnet_int     | Fully connected neural network benchmark.        | [Paper](https://arxiv.org/abs/1905.04970), [GitHub](https://github.com/automl/nas_benchmarks)        |
| HPO-B         |    hpob, hpob_int    | Hyperparameter optimization benchmark suite.     | [Paper](https://arxiv.org/abs/2106.06257), [GitHub](https://github.com/automl/HPOBench)              |


The "int"-prefix stands for interactive. 

The following optimizers are implemented:
| Optimizer     | Option Name | URL to Paper/GitHub                                                                 | Note |
|--------------|------------|------------------------------------------------------------------------------------|------|
| IBO-HPC      | pc         | this GitHub                                                                      |      |
| Optuna       | optunabo   | [Website](https://optuna.org/), [Documentation](https://optuna.readthedocs.io/en/stable/) | implements "BO w/ TPE" in paper   |
| skopt        | skopt      | [GitHub](https://github.com/scikit-optimize/scikit-optimize)                      |  implements "BO w/ RF" in paper    |
| PiBO         | pibo       | [Paper](https://openreview.net/pdf?id=MMAeCXIa89)                                 |      |
| BOPrO        | bopro      | [Paper](https://arxiv.org/pdf/2006.14608)                                         |      |
| Local Search | ls        | -                                                                                  |      |
| SMAC         | smac       | [GitHub](https://github.com/automl/SMAC3)                                         |      |
| PriorBand    | -         | [Paper](https://arxiv.org/pdf/2306.12370)                                         |      |
| Random Search | rs        | -                                                                                  |      |


> Note: PriorBand is currently not supported in the Docker environment. However, it can be started on the host machine (which might trigger some errors due to missing 
dependencies). The reason is that it is currently incompatible with packages installed in the Docker container. Will be fixed in future releases.

## Interactions
To simulate user interactions we allow you to put json-files into the `interventions` directory. The json-files should have the following structure:

```
[
    {
        "type": "good" or "bad",
        "kind": "point" or "dist"
        "intervention": {YOUR INTERVENTION},
        "iteration" ITERATION TO TRIGGER INTERACTION
    },
    ...
]
```

You can put as many interactions as you like into the json-file. The interactions will be evaluated top down, i.e. the first appearing in the list first and so on.
The `iteration` field only holds information for the optimizer when to use the interaction but it has no effect on the order of how the interactions are triggered. This means that **interactions have to be ordered correctly in the list!** Otherwise this might cause errors or unexpected behavior.

The "type" is only meta-data and specifies whether an interaction is intended to be beneficial or harmful. The "kind" field defines whether the interaction is a point-interaction, i.e. the user fixes a certain hyperparameter to a certain value or whether the interaction is a distribution, i.e. the user specifies a distribution over possible interactions. The "intervention" field then specifies the exact interaction as a dictionary where {"hyperparameter": VALUE or {"dist:" ["uniform, cat, gauss"], "parameters": [parameters]}}" is the syntax. The "iteration" field specifies the iteration the interaction is applied. See the existing json-files in the `interventions`-directory for examples.

## Troubleshooting
Sometimes you might encounter an error of scikit-learn stating that some metric(s) cannot be imported.
There is a hacky workaround that resolves this issue:

1. `pip uninstall scikit-learn pandas`
2. `pip install scikit-learn pandas`
3. `pip uninstall scikit-learn`
4. `pip install scikit-learn==1.1.3`

This forces pip to downgrade to `scikit-learn==1.1.3.` while keeping the other packages at higher versions (might cause warnings by pip). The code should work after that.

# Plotting
To reproduce the plots from the paper, please follow the instructions below:
1. Download the raw log data from [this URL](https://figshare.com/ndownloader/files/53323751): `wget -O ./data.zip https://figshare.com/ndownloader/files/53323751`
2. If it does not exist, create the directory `/app/data`
3. Extract the raw data to `/app/data`: `unzip ./data.zip -d /app/data/`
4. Navigate to `/app/discombo/utils`
5. Execute `python plot.py --exp [BENCHMARK] --out-file plot.pdf --with-std` where `BENCHMARK` is one of the benchmark options from the table above (column "Option Name") 

# License
To this repository, the CC BY-NC 4.0 license applies.