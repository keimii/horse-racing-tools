import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import cm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def num_vis(data, beta):
    bins = int(data.shape[0] / 100)
    data.hist(figsize=(5, 4), color="blue", bins=bins, alpha=0.4)
    mean = data.mean()
    ymax = pd.cut(data, bins).value_counts().max()
    plt.vlines(
        x=mean, ymin=0, ymax=ymax, colors="red", linestyles="--", lw=1
    )  # 平均値の直線追加
    plt.annotate(
        "Mean of prova: {}".format(round(mean, 2)), xy=(mean, ymax), color="red"
    )
    plt.vlines(
        x=beta, ymin=0, ymax=ymax, colors="darkgreen", linestyles="--", lw=1
    )  # 正当値の直線追加
    plt.annotate(
        "Rate of correct value: {}".format(round(beta, 2)),
        xy=(beta, ymax * 0.8),
        color="darkgreen",
    )
    plt.title(data.name)
    plt.show()


def plot_confusion_matrix(test_y, pred_y, classes, normalize=False, figsize=10):
    cm = confusion_matrix(test_y, pred_y)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=[figsize, figsize])
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label\n",
        xlabel="\nPredicted label",
    )
    fmt = ".2f" if normalize else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                size=20,
                color="black",
                bbox=dict(facecolor="white", alpha=0.7),
            )
    fig.tight_layout()

    plt.show()


def plot_accurency_summary(val_label, pred_labels):
    accurency = accuracy_score(val_label, pred_labels)
    recall = recall_score(val_label, pred_labels, average="macro")
    precision = precision_score(val_label, pred_labels, average="macro")
    f1 = f1_score(val_label, pred_labels, average="macro")

    cv_table = pd.DataFrame(
        [[accurency], [recall], [precision], [f1]],
        index=["正解率 (Accuracy)", "網羅率 (Recall)", "的中率 (Precision)", "F 値"],
        columns=["値"],
    )
    display(cv_table)


def goal_rank_hist(data):
    bins = 18
    data.hist(figsize=(5, 4), color="darkblue", bins=bins, alpha=0.7)
    mean = data.mean()
    median = data.median()
    ymax = pd.cut(data, bins).value_counts().max()
    plt.vlines(
        x=mean, ymin=0, ymax=ymax, colors="red", linestyles="--", lw=0.7
    )  # 平均値の直線追加
    plt.annotate("Mean: {}".format(round(mean, 2)), xy=(mean, ymax), color="red")
    plt.vlines(
        x=median, ymin=0, ymax=ymax, colors="orange", linestyles="--", lw=0.7
    )  # 中央値の直線追加
    plt.annotate(
        "Median: {}".format(round(median, 2)), xy=(mean, ymax * 0.8), color="orange"
    )
    plt.title(data.name)
    plt.show()


def plot_loss_curve(train_logloss, test_logloss):
    fig, axs = plt.subplots(figsize=[10, 5])

    axs.plot(train_logloss, label="Train")
    axs.plot(test_logloss, label="Test")
    axs.set_ylabel("Log loss")
    axs.set_xlabel("Boosting round")
    axs.set_title("Training performance")
    axs.legend()

    plt.xticks(rotation=80)
    plt.show()


def plot_calibration_curve(plot_df, predict_name, label_name):
    # 各datasetのlgbによるtestデータのcailbration curveをまとめてかく
    # Plot calibration plots
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    print(
        f"{label_name} AUC: {roc_auc_score(plot_df[label_name], plot_df[predict_name])}"
    )
    fraction_of_positives, mean_predicted_value = calibration_curve(
        plot_df[label_name], plot_df[predict_name], n_bins=10
    )
    ax1.plot(
        mean_predicted_value, fraction_of_positives, "s-", label="%s" % (label_name,)
    )
    ax2.hist(
        plot_df[predict_name],
        range=(0, 1),
        bins=10,
        label=label_name,
        histtype="step",
        lw=2,
    )

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plots  (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


def plot_correlation_graph(df):
    rel_df = df[["proba", "payoff", "odds"]]
    rel_df["hit"] = (rel_df["payoff"] > 0) * 1
    rel_df["proba_hist"] = (rel_df["proba"] * 100).apply(np.floor) / 100
    quinella_grouped = rel_df.groupby("proba_hist").agg(
        {"hit": np.mean, "odds": np.median}
    )

    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.scatter(quinella_grouped.index, quinella_grouped.hit)
    plt.subplot(132)
    plt.scatter(quinella_grouped.index, quinella_grouped.odds)


def plot_income_graph(graph_df):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # 1つのaxesオブジェクトのlines属性に2つのLine2Dオブジェクトを追加
    ax1.bar(graph_df.index, graph_df["income"], color=cm.Set1.colors[1], label="income")
    ax2.plot(
        graph_df.index,
        graph_df["total_income"],
        color=cm.Set1.colors[0],
        label="total_income",
    )

    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    ax1.legend(handler1 + handler2, label1 + label2, loc=2, borderaxespad=0.0)

    income_max = 1.2 * max(graph_df["income"])
    income_min = 1.2 * min(graph_df["income"])
    total_income_max = 1.2 * max(graph_df["total_income"])
    total_income_min = 1.2 * min(graph_df["total_income"])

    ax1.set_ylim([income_min, income_max])
    ax2.set_ylim([total_income_min, total_income_max])
