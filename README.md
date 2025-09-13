# Active Pusher

Implementation of paper "ActivePusher: Active Learning and Planning with Residual Physics for Nonprehensile Manipulation".


## Dependencies
This project is developed in Ubuntu 22.04 with python 3.10.

### Installation of dependencies
```bash
pip install -r requirements.txt
```

Install pytorch following the instructions from [pytorch.org](https://pytorch.org/get-started/locally/).

Install OMPL python bindings with provided wheels [OMPL Github Releases](https://github.com/ompl/ompl/releases).


## ActivePusher in Simulation
In this project, we use [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main) robotic simulation as it supports parallelized simulation for fast data collection and evaluation. 

### Launch Simulation
Our simulation environment is implemented in `sim.py`, and we also implement a server-client structure to have the simulation running in a different process. To launch the simulation server, run
```bash
python sim_network.py
```
You should see the simulation environment shown up with N instances of the same scene.

### Collect Data
To collect push data of a given object, run
```bash
python collect_data.py
```
You should see the robot start to push the object randomly. The push parameter and the effect of the push will be recorded and saved in **data** folder.

### Run Active Training
After data collection, you can train your model directly with `train_model.py` with specified parameters. 

Or you can run the full the full active learning pipeline with `train_active.py`.
```bash
python train_active.py <object_name> <model_class>
```
where `<object_name>` is the name of the object to be trained (e.g. `cracker_box`), and `<model_class>` is the class of the model to be used, either `residual` or `mlp`.

There are other optional arguments to specify the number of experiments, the number of queries per loop, and the random seed, etc.

After training, all the models will be saved in **results/models** folder and all the historical data will be saved in **results/learning** folder.

### Run Planning
To execute a plan-to-edge motion planning task (currently only support `cracker_box`), run
```bash
python push_to_edge.py <iteration> <model_name> <sampling>
```
where `<iteration>` is the iteration number (this is designed for parallelization. If you want to run once only, simply use `0`), `<model_name>` is the name of the model to be used (defined as `${obj_name}_${model_selection}_${exp_idx}_${num_data}` in `parallelization/run_planning.sh`), and `<sampling>` is whether to use active sampling.

Since we can leverage the parallelization for planning, we provide a shell script to run the planning tasks in parallel. Remember to run the simulation server in another thread first since this script also tests the plans right away.
```bash
bash parallelization/run_planning.sh <num_threads> <num_repeats>
```
where `<num_threads>` is the number of threads to run the planning tasks in parallel, and `<num_repeats>` is the number of times to repeat the planning tasks. By default, each thread tries to solve 10 pre-defined planning tasks.

### Plot Results
After training or planning, you can plot the corresponding results with `plot_active.py` and `plot_planning.py`. 
For active learning, run
```bash
python plot_active.py
```
For planning, run
```bash
python plot_planning.py
```

### Known Simulation Issues
Genesis adapts pyramid friction cone model for contact, which causes the overall environment non-isotropic, violating the assumption of our paper. Therefore, when collecting data and executing planning, we always reset the object to a pre-defined initial state.
