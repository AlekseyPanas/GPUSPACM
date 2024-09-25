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
    if "npdat" not in os.listdir(LOG_ROOT_PATH):
        os.mkdir(NPY_LOG_PATH)

    file_idx = choose_option_from_list(os.listdir(NPY_LOG_PATH))

    filepath = os.path.join(NPY_LOG_PATH, os.listdir("../npdat")[file_idx])
    reader = NumpyDataReader(filepath)

    options = ["Text", "Matplotlib", "Tensorboard (w/ rollback)", "Tensorboard (w/o rollback)"]
    opt_idx = choose_option_from_list(options)

    if opt_idx == 0:
        TextConverter(reader, "textlogs", LOG_ROOT_PATH,
                      (lambda :(print("Ignore zero-force events in log?"), choose_option_from_list(["y", "n"]))[1])() == 0,
                      (lambda :(print("Ignore rollbacks in log?"), choose_option_from_list(["y", "n"]))[1])() == 0,
                      (lambda :(print("Output in binary?"), choose_option_from_list(["y", "n"]))[1])() == 0,
                      (lambda: (print("Print progress?"), choose_option_from_list(["y", "n"]))[1])() == 0
                      ).convert()
    elif opt_idx == 1:
        MatplotlibConverter(reader, "matplotlibplots", LOG_ROOT_PATH).convert()
    elif opt_idx == 2:
        TensorboardConverter(reader, "runs", LOG_ROOT_PATH, False).convert()
    elif opt_idx == 3:
        TensorboardConverter(reader, "runs", LOG_ROOT_PATH, True).convert()
    else:
        print("Sus.... this codepath should be impossible")