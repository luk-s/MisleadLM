from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import wandb


def get_rewards_and_accuracies(
    run_name: str,
    reward_metric: str = "metrics/preference reward",
    accuracy_metric: str = "metrics/acc",
) -> Tuple[List[float], List[float]]:
    """
    Retrieve reward and accuracy metrics from a Weights & Biases run.

    Args:
        run_name (str): The name of the W&B run to retrieve data from
        reward_metric (str, optional): The name of the reward metric in W&B.
            Defaults to "metrics/preference reward".
        accuracy_metric (str, optional): The name of the accuracy metric in W&B.
            Defaults to "metrics/acc".

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists:
            - rewards: List of reward values from the run
            - accuracies: List of accuracy values from the run
    """
    api = wandb.Api()
    run = api.run(run_name)
    history = run.scan_history(keys=[reward_metric, accuracy_metric])
    rewards = [row[reward_metric] for row in history]
    accuracies = [row[accuracy_metric] for row in history]
    return rewards, accuracies


def plot_rewards_and_accuracies(rewards: List[float], accuracies: List[float]) -> None:
    """
    Create and save a bar chart comparing initial and maximum rewards and accuracies.

    The plot shows two groups of bars:
    1. Reward values (R^train) - comparing initial policy vs RLHF policy
    2. Accuracy values (R*) - comparing initial policy vs RLHF policy

    Args:
        rewards (List[float]): List of reward values from training
        accuracies (List[float]): List of accuracy values from training

    Returns:
        None: The function saves the plot to "rewards_and_accuracies.png"
    """
    # Increase the global font size
    plt.rcParams.update({"font.size": 16})

    # If the accuracies are provided in the interval [0, 1], convert them to the interval [0, 100]
    if max(accuracies) <= 1:
        accuracies = [acc * 100 for acc in accuracies]

    # Find the position of the maximum reward
    max_reward_idx = np.argmax(rewards)

    # Extract the first and max reward values
    first_reward = rewards[0]
    max_reward = rewards[max_reward_idx]

    # Extract the first accuracy and accuracy at max reward
    first_accuracy = accuracies[0]
    max_accuracy = accuracies[max_reward_idx]

    # Create figure and axes
    fig, ax2 = plt.subplots(figsize=(6, 6))
    ax1 = ax2.twinx()

    # Set width of bars
    bar_width = 0.3
    offset = 0.5

    # Set positions for bars - reversed from previous version
    r1 = np.array([0, offset + bar_width])  # Positions for first bars in each group
    r2 = np.array(
        [bar_width, offset + 2 * bar_width]
    )  # Positions for second bars in each group

    # Define colors using RGB values
    blue_color = (113 / 255, 193 / 255, 209 / 255)  # (113,193,209)
    orange_color = (241 / 255, 180 / 255, 90 / 255)  # (241,180,90)

    # Create bars - reversed from previous version (reward on left, accuracy on right)
    # Reward bars on the left (ax1)
    bottom = -2.0
    ax1.bar(
        r1[0],
        first_reward - bottom,
        width=bar_width,
        color=blue_color,
        label="$π_{init}$",
        bottom=bottom,
    )
    ax1.bar(
        r2[0],
        max_reward - bottom,
        width=bar_width,
        color=orange_color,
        label="$π_{rlhf}$",
        bottom=bottom,
    )

    # Accuracy bars on the right (ax2)
    ax2.bar(r1[1], first_accuracy, width=bar_width, color=blue_color)
    ax2.bar(r2[1], max_accuracy, width=bar_width, color=orange_color)

    # Add labels and title
    ax1.set_xlabel("Metrics")
    ax1.set_ylabel("Reward (for $R^{train}$)")
    ax2.set_ylabel("Accuracy (for $R^*$)")

    # Set x-ticks with proper font size
    ax2.set_xticks([bar_width / 2, offset + 1.5 * bar_width])
    ax2.set_xticklabels(["$R^{train}$", "$R^*$"], fontsize=24)

    # Set axis limits
    ax1.set_ylim(bottom, 1.0)
    ax2.set_ylim(40, 70)

    # Add legend
    ax1.legend(loc="upper right")

    # Adjust layout
    fig.tight_layout()

    # Save plot
    plt.savefig("rewards_and_accuracies.png", dpi=300)


if __name__ == "__main__":
    RUN_NAME = "lukas-fluri/trlx/r54glzw7"
    rewards, accuracies = get_rewards_and_accuracies(run_name=RUN_NAME)

    plot_rewards_and_accuracies(rewards, accuracies)
