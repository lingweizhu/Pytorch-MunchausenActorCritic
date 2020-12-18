import os
import argparse
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import json
matplotlib.use("Agg")


def get_results(exp_path, y_label):
    def read_scalars(resultpath):
        # read scalars from event file
        for p in resultpath.rglob("events*"):
            eventspath = p
        event_acc = event_accumulator.EventAccumulator(
            str(eventspath), size_guidance={'scalars': 0})
        event_acc.Reload()

        scalars = {}
        steps = {}

        for tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(tag)
            start_time = events[0].wall_time
            scalars[tag] = [event.value for event in events]
            steps[tag] = [event.step for event in events]
            # for training minutes steps
            min_tag = tag.split("/")[:-1] + ["minutes"]
            scalars["/".join(min_tag)] = [event.value for event in events]
            steps["/".join(min_tag)] = [(event.wall_time - start_time) / 60
                                        for event in events]

        return steps, scalars

    def get_return_dataframe(steps, scalars, y_label):
        # convert steps and scalars to dataframe
        df_dict = {}
        for key in steps.keys():
            step = key.split("/")[-1]
            if step != y_label:
                continue
            dicimal = -2
            df_dict[step] = pd.DataFrame(
                data={"iterations": np.round(steps[key], dicimal),
                      y_label: scalars[key]})

        return pd.concat(df_dict, axis=1)

    results = defaultdict(lambda: defaultdict(lambda: []))
    for env in exp_path.glob("[!.]*[!.png]"):
        for result in env.glob("[!.]*"):
            agent = result.name
            for seed in result.glob("[!.]*"):
                # load result
                steps, scalars = read_scalars(seed)
                try:
                    df = get_return_dataframe(steps, scalars, y_label=y_label)
                except ValueError:
                    print(str(result) + " doesn't have data.")
                    continue
                results[env.name][agent].append(df)

        # concatenate same agent
        for agent in results[env.name].keys():
            results[env.name][agent] = \
                pd.concat(results[env.name][agent])
    return results 


def plot(exp_path, y_label):
    exp_path = Path(exp_path)
    results = get_results(exp_path, y_label)
    # layout
    x = "iterations"

    num_cols = len(results)
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols*6, 4))
    if num_cols == 1:
        axes = [axes]
    sns.set(style="darkgrid")

    # agent colors
    agents = []
    for env in results:
        for agent in results[env]:
            if agent not in agents:
                agents.append(agent)

    colors = sns.color_palette(n_colors=len(agents))

    for i, env in enumerate(results):
        xlim = 0
        for agent in results[env]:
            # plot results
            df = results[env][agent][y_label]
            sns.lineplot(x=x,
                         y=y_label,
                         ci="sd",
                         data=df,
                         ax=axes[i],
                         label=agent,
                         legend=None,
                         color=colors[agents.index(agent)])
            xlim = max(xlim, df[x].max())

        axes[i].set_title(env)
        axes[i].set_xlim(0, xlim)

    handles = [None] * len(agents)

    for ax in axes:
        handle, label = ax.get_legend_handles_labels()
        for h, agent in zip(handle, label):
            handles[agents.index(agent)] = h

    lgd = fig.legend(handles, agents, loc="upper center",
                     bbox_to_anchor=(0.5, 1.1), ncol=len(agents))
    fig.tight_layout()
    fig.savefig(str(exp_path / (y_label+".png")),
                bbox_extra_artists=(lgd, ), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots the results of experiments.")
    parser.add_argument("dir")
    parser.add_argument("-y", default="test")
    args = parser.parse_args()

    plot(args.dir, y_label=args.y)
