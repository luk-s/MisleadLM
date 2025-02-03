from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import ast

EVAL_DATA = [
    {"eval_loss": 0.6602340936660767, "eval_accuracy": 0.6243997404282933, "epoch": 0.05},
    {"eval_loss": 0.6402640342712402, "eval_accuracy": 0.6375081116158339, "epoch": 0.1},
    {"eval_loss": 0.6372454762458801, "eval_accuracy": 0.6356911096690461, "epoch": 0.16},
    {"eval_loss": 0.6246324181556702, "eval_accuracy": 0.6582738481505516, "epoch": 0.21},
    {"eval_loss": 0.6404783725738525, "eval_accuracy": 0.6507462686567164, "epoch": 0.26},
    {"eval_loss": 0.6201765537261963, "eval_accuracy": 0.6561972744970798, "epoch": 0.31},
    {"eval_loss": 0.6152067184448242, "eval_accuracy": 0.6724205061648281, "epoch": 0.36},
    {"eval_loss": 0.6128213405609131, "eval_accuracy": 0.6755353666450357, "epoch": 0.41},
    {"eval_loss": 0.6121490001678467, "eval_accuracy": 0.6781310837118754, "epoch": 0.47},
    {"eval_loss": 0.626394510269165, "eval_accuracy": 0.6699545749513303, "epoch": 0.52},
    {"eval_loss": 0.6088222861289978, "eval_accuracy": 0.6798182998053213, "epoch": 0.57},
    {"eval_loss": 0.6104105114936829, "eval_accuracy": 0.6683971447112265, "epoch": 0.62},
    {"eval_loss": 0.6154842376708984, "eval_accuracy": 0.673069435431538, "epoch": 0.67},
    {"eval_loss": 0.6065412759780884, "eval_accuracy": 0.6803374432186892, "epoch": 0.73},
    {"eval_loss": 0.6184839010238647, "eval_accuracy": 0.6693056456846204, "epoch": 0.78},
    {"eval_loss": 0.6075698733329773, "eval_accuracy": 0.6765736534717716, "epoch": 0.83},
    {"eval_loss": 0.6099688410758972, "eval_accuracy": 0.6735885788449059, "epoch": 0.88},
    {"eval_loss": 0.607663094997406, "eval_accuracy": 0.6716417910447762, "epoch": 0.93},
    {"eval_loss": 0.6079190969467163, "eval_accuracy": 0.6715120051914342, "epoch": 0.99},
    {"eval_loss": 0.7143658399581909, "eval_accuracy": 0.6704737183646983, "epoch": 1.04},
    {"eval_loss": 0.7500689625740051, "eval_accuracy": 0.6660609993510708, "epoch": 1.09},
    {"eval_loss": 0.7477913498878479, "eval_accuracy": 0.663854639844257, "epoch": 1.14},
]
TRAIN_DATA = [
    {"loss": 0.689, "grad_norm": 1.337977409362793, "learning_rate": 9.896265560165976e-06, "epoch": 0.05},
    {"loss": 0.6438, "grad_norm": 1.544917345046997, "learning_rate": 9.792531120331951e-06, "epoch": 0.1},
    {"loss": 0.6447, "grad_norm": 5.714319229125977, "learning_rate": 9.688796680497927e-06, "epoch": 0.16},
    {"loss": 0.6453, "grad_norm": 1.8109767436981201, "learning_rate": 9.585062240663902e-06, "epoch": 0.21},
    {"loss": 0.6188, "grad_norm": 3.3889307975769043, "learning_rate": 9.481327800829876e-06, "epoch": 0.26},
    {"loss": 0.6177, "grad_norm": 3.698004722595215, "learning_rate": 9.377593360995851e-06, "epoch": 0.31},
    {"loss": 0.623, "grad_norm": 1.8468977212905884, "learning_rate": 9.273858921161826e-06, "epoch": 0.36},
    {"loss": 0.6097, "grad_norm": 1.0102955102920532, "learning_rate": 9.170124481327802e-06, "epoch": 0.41},
    {"loss": 0.6273, "grad_norm": 1.6054259538650513, "learning_rate": 9.066390041493777e-06, "epoch": 0.47},
    {"loss": 0.6119, "grad_norm": 1.5514922142028809, "learning_rate": 8.962655601659752e-06, "epoch": 0.52},
    {"loss": 0.6138, "grad_norm": 1.2349945306777954, "learning_rate": 8.858921161825726e-06, "epoch": 0.57},
    {"loss": 0.6035, "grad_norm": 1.4817817211151123, "learning_rate": 8.755186721991701e-06, "epoch": 0.62},
    {"loss": 0.6175, "grad_norm": 1.2530572414398193, "learning_rate": 8.651452282157678e-06, "epoch": 0.67},
    {"loss": 0.6189, "grad_norm": 1.081425666809082, "learning_rate": 8.547717842323652e-06, "epoch": 0.73},
    {"loss": 0.605, "grad_norm": 2.826841115951538, "learning_rate": 8.443983402489627e-06, "epoch": 0.78},
    {"loss": 0.6142, "grad_norm": 0.7951761484146118, "learning_rate": 8.340248962655602e-06, "epoch": 0.83},
    {"loss": 0.61, "grad_norm": 1.1533509492874146, "learning_rate": 8.236514522821578e-06, "epoch": 0.88},
    {"loss": 0.6065, "grad_norm": 2.176241397857666, "learning_rate": 8.132780082987553e-06, "epoch": 0.93},
    {"loss": 0.6051, "grad_norm": 0.8070855736732483, "learning_rate": 8.029045643153528e-06, "epoch": 0.99},
    {"loss": 0.4985, "grad_norm": 2.6110825538635254, "learning_rate": 7.925311203319502e-06, "epoch": 1.04},
    {"loss": 0.3828, "grad_norm": 5.983684062957764, "learning_rate": 7.821576763485477e-06, "epoch": 1.09},
    {"loss": 0.3662, "grad_norm": 5.13045597076416, "learning_rate": 7.717842323651453e-06, "epoch": 1.14},
]

LOG_FILE_PATH = "scripts/outputs/slurm-744667.out"


def load_data(log_file_path: str) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    train_data = []
    eval_data = []
    with open(log_file_path, "r") as file:
        for line in file:
            if "eval_loss" in line:
                data = line[line.index("{") : line.index("}") + 1]
                eval_data.append(ast.literal_eval(data))
            elif "learning_rate" in line:
                data = line[line.index("{") : line.index("}") + 1]
                train_data.append(ast.literal_eval(data))

    return train_data, eval_data


def plot_progress(
    eval_epochs: List[float], eval_accuracies: List[float], train_epochs: List[float], losses: List[float]
) -> None:
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="red")
    ax1.plot(train_epochs, losses, color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Adding Twin Axes

    ax2 = ax1.twinx()

    ax2.set_ylabel("Evaluation accuracy (%)", color="blue")
    ax2.plot(eval_epochs, eval_accuracies, color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Show plot

    plt.savefig(f"training_progress_epoch_{max(train_epochs[-1],eval_epochs[-1])}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    train_data, eval_data = load_data(LOG_FILE_PATH)

    eval_epochs = [dic["epoch"] for dic in eval_data]
    eval_accuracies = [100 * dic["eval_accuracy"] for dic in eval_data]
    train_epochs = [dic["epoch"] for dic in train_data]
    losses = [dic["loss"] for dic in train_data]

    plot_progress(eval_epochs=eval_epochs, eval_accuracies=eval_accuracies, train_epochs=train_epochs, losses=losses)
