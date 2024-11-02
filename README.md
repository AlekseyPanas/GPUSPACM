# GPUSPACM
[Speculative Parallel Asynchronous Contact Mechanics (SPACM)](https://www.cs.columbia.edu/cg/spacm/rollback.pdf) is an algorithm that solves contact mechanics in physics simulations. SPACM guarantees that no interpenetrations can ever occur while also maintaining good energy behavior (i.e symplecticity: energy does not grow or dissipate). Collisions in SPACM are prevented via penalty layers, which are regions around an object that exert a repelling force on anything that gets near. There can be up to infinity of such layers on an object with the key idea that each subsequent layer is only activated if needed: the simulation runs for some short window and detects if collisions were missed, in which case it rolls back and activates another repelling layer until no interpenetration occurs.

Processing a penalty force event involves looping through every object in the simulation, computing the penalty force on that object (based on the object's postion, i.e if it's within the region), and integrating the force via a symplectic time integrator. Because each particle is evaluated independently, SPACM is highly parallelizable! As such, our goal is to implement SPACM on the GPU using CUDA, leveraging this parallelizability for major performance improvments.

## Usage
- The `SPACM1DSim` class under `z_sims/OneDimensionalSPACM.py` provides the 1D SPACM simulation. The `run_sim` method can be called on this class to return an iterable generator of the simulation
- `SPACM1DSim` requires a logger. Create and pass an instance of `NumpyLogger` found under `z_logging/loggers.py`. This will generate `.npy` binary numpy array files containing simulation event logs
- To run a simulation, you need an "executor". All executors can be found under the `z_executors` directory. Each executor is run by instantiating it with an instance of a simulation class (i.e `SPACM1DSim`) and calling the `.run()` method. The following executors exist:
  - `PygameVisualizer` under `z_executors/pygame_visualizer.py` launches an interactive pygame window allowing you to visualize many aspects of the simulation. Common controls are shown in the top right corner of this window. Use mouse wheel to zoom in and out. Drag with mouse to move the "camera". Hold shift and scroll/drag to scale the timeline (which shows force events as they're being processed)
  - `NoGuiExecutor` under `z_executors/nogui_executor.py` runs the sim in the terminal with no interaction or visualization. Use this to run the sim as quickly as possible. To configure when the sim should end, call one of the setter methods in the class prior to calling `.run()` (look into the class to see what setters exist and what they do)
- `sim_main.py` contains example code for running the sim, covering most of the above bullet points
- `log_generator_main.py` launches an interactive terminal script for converting `.npy` log files into various readable formats: (Tensorboard, matplotlib, text). Run this file and it should automatically prompt you
- If you encounter an `ImportError` anywhere, just google how to pip install the missing library and try again. We depend on tensorboard, pygame, numpy, matplotlib, and probably some others I'm forgetting
- If you still encounter errors, make sure you are in Python3.11+. We use type hints and syntax which doesn't exist in earlier versions
- If you still encounter errors, then either I am currently working on the code and it doesn't work yet, you've found a bug that I didn't know about, or something is wrong on your end

## Progress
- [x] Implement SPACM in 1D
- [ ] Implement SPACM in 2D
- [ ] Implement SPACM in 3D
- [ ] Implement SPACM on the GPU using warp
- [ ] Implement SPACM for rigid bodies
