from __future__ import annotations
from z_sims.OneDimensionalSPACM import SPACM1DSim
from z_sims.OneDimensionalSPACMHardCodedLayers import SPACM1DSimHardcodedLayers
from z_logging.loggers import NumpyLogger
from z_executors.pygame_visualizer import PygameVisualizer
from z_executors.nogui_executor import NoGuiExecutor
import os

LOG_ROOT_PATH = os.getcwd()


if __name__ == "__main__":
    # Logging setup
    do_log = False

    # sim = SPACM1DSim(0.03, -1, 0.005, [0.3], [0], [0], [1], 1, 0.5, 0.005, NumpyLogger(LOG_ROOT_PATH, do_log, "SingleBallStartOnLayer"))
    # sim = SPACM1DSim(0.03, -1, 0.005, [3], [0], [0], [1], 1, 0.5, 0.005, NumpyLogger(LOG_ROOT_PATH, do_log, "SingleBallWithRollbackDissipationTest-PostHeapifyFix"))
    # sim = SPACM1DSimHardcodedLayers(0.03, -1, 0.005, [3], [0], [0], [1], 1, 0.5, 0.005, NumpyLogger(LOG_ROOT_PATH, do_log, "SingleBallHardcodedLayersDissipationTest"))
    # sim = SPACM1DSim(30, -1, 0.005, [3], [0], [0], [1], 1, 0.5, 0.005, NumpyLogger(LOG_ROOT_PATH, do_log, "SingleBallWithRollbackDissipationTestHugeWindow"))

    # sim = SPACM1DSim(0.03, -1, 0.01, [3, 5], [0, 1], [0], [1, 1], 1, 1, 0.01, NumpyLogger(LOG_ROOT_PATH, do_log, "TwoBallHardcodedLayersDissipationTest-PostHeapifyFix"))  # Working two-particle sim, but bounces are too far
    sim = SPACM1DSim(0.03, -1, 0.01, [3, 5, 7], [0, 1, 2], [0, 8.3], [1, 1, 1], 1, 1, 0.01, NumpyLogger(LOG_ROOT_PATH, do_log, "ThreeBallHardcodedLayersDissipationTest-PostHeapifyFix"))  # Working three-particle two-wall sim, but bounces are too far

    # sim = SPACM1DSim(0.03, -1, 0.001, [3, 5, 7], [0, 1, 2], [0], [1, 1, 1], 4, 0.3, 0.001)  # Working two-particle sim, bounces not far, but is slow
    # sim = SPACM1DSim(0.03, -1, 0.001, [3, 5, 7], [0, 1, 2], [0, 8], [1, 1, 1], 10, 0.4, 0.001)  # Working three-particle two-wall sim, accurate but slow

    # sim = SPACM1DSim(0.03, -1, 0.01, [10], [0], [0], [1], 1, 0.1, 0.01)  # Infinite loop
    # sim = SPACM1DSim(0.03, -1, 0.03, [10], [0], [0], [1], 1, 0.05, 0.03)  # Infinite loop

    visualizer = PygameVisualizer((800, 800), sim, 1.1, 0.005)
    visualizer.run()

    #NoGuiExecutor(sim).set_mode_sim_time(1000).run()
