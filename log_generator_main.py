import os
from z_logging.converters import NumpyDataReader, MatplotlibConverter, TextConverter, TensorboardConverter
from z_logging.loggers import NumpyLogger

LOG_ROOT_PATH = os.getcwd()
NPY_LOG_PATH = os.path.join(LOG_ROOT_PATH, NumpyLogger.FOLDER_NAME)


def choose_option_from_list(opts: list[str]) -> int:
    print("\n".join(f"[{idx}] {opt}" for idx, opt in enumerate(opts)))
    print("=============")
    while True:
        try:
            opt_idx = int(input("Select an option: ").strip())
            if opt_idx >= len(opts) or opt_idx < 0:
                print("Bruh that ain't one of the given options dawg")
            else:
                break
        except Exception:
            print("Bruh that ain't a number dawg")
    return opt_idx


if __name__ == "__main__":
    if NumpyLogger.FOLDER_NAME not in os.listdir(LOG_ROOT_PATH):
        print("No Logs Found")
        exit(0)

    subfolder_idx = choose_option_from_list(os.listdir(NPY_LOG_PATH))

    subfolder_path = os.path.join(NPY_LOG_PATH, os.listdir(NPY_LOG_PATH)[subfolder_idx])
    reader = NumpyDataReader(subfolder_path)

    options = ["Text", "Matplotlib", "Tensorboard"]
    opt_idx = choose_option_from_list(options)

    if opt_idx == 0:
        TextConverter(reader, "textlogs", LOG_ROOT_PATH,
                      (lambda :(print("Ignore zero-force events in log?"), choose_option_from_list(["y", "n"]))[1])() == 0,
                      (lambda :(print("Ignore rollbacks in log?"), choose_option_from_list(["y", "n"]))[1])() == 0,
                      (lambda :(print("Output in binary?"), choose_option_from_list(["y", "n"]))[1])() == 0,
                      (lambda: (print("Print progress?"), choose_option_from_list(["y", "n"]))[1])() == 0
                      ).convert()
    elif opt_idx == 1:
        MatplotlibConverter(reader, "matplotlibplots", LOG_ROOT_PATH,
                            (lambda: (print("Only Energy?"), choose_option_from_list(["y", "n"]))[1])() == 0).convert()
    elif opt_idx == 2:
        TensorboardConverter(reader, "runs", LOG_ROOT_PATH).convert()
    else:
        print("Sus.... this codepath should be impossible")