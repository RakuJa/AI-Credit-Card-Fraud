import polars as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from polars import DataFrame
from scipy.stats import pearsonr

from utils import print_separator


def explore_dataset(
    df: DataFrame, show_graphs: bool = False, save_graphs: bool = False
) -> DataFrame:
    # Start by cleaning up, making everything lower case & converting data types
    df.columns = [col.lower() for col in df.columns]
    df.replace_column(
        df.get_column_index("class"), df.get_column("class").cast(pd.String)
    )

    print(df.head())
    print(df.shape)

    # Non-null count for all columns
    print("Non-null count for all columns")
    print((df.null_count() * -1) + pd.Series("", [df.shape[0]]))
    # Null count for all columns
    print("Null count for all columns")
    print(df.null_count())
    # Percentage of unique values
    print("Percentage of unique values")
    print(df["class"].value_counts(normalize=True))

    print_separator()

    # Plot
    f, ax = plt.subplots(figsize=(8, 10))
    _ax = sns.countplot(x="class", data=df, hue="class", legend=False)

    if save_graphs:
        plt.savefig("images/dataset/imbalanced_dataset.png", transparent=True)

    # Describes time and amount values
    print(df[["time", "amount"]].describe())
    # On average, each transaction happens every time:mean seconds
    # This two columns have outliers. It can be seen from the difference w max value

    # Get data about fraud transaction vs normal transactions (count, columns)
    fraud = df.filter(pd.col("class") == "1")
    normal = df.filter(pd.col("class") == "0")
    print(f"Shape of Fraud Transactions: {fraud.shape}")
    print(f"Shape of Normal Transactions: {normal.shape}")

    print_separator()

    # Compare side by side the fraud and normal amounts
    # description

    n = normal["amount"].describe()
    # Combine horizontally
    print(
        pd.concat(
            [
                n.select("statistic"),
                fraud["amount"]
                .describe()
                .select("value")
                .rename({"value": "fraud_amount"}),
                n.select("value").rename({"value": "normal_amount"}),
            ],
            how="horizontal",
        )
    )
    # Check the monetary amount involved in frauds
    print(fraud["amount"].value_counts(sort=True).head())
    #  With fraud transactions, the average amount of fraud is 122.22 USD,
    #  the highest is 2125 USD,
    #  the lowest is 0 and
    #  the maximum amount is 1 USD with 113 times.

    plt.figure(figsize=(8, 6))
    plt.title("Distribution of Transaction Time", fontsize=14)
    sns.histplot(df["time"], bins=100)

    # This data set contains two-day trading information,
    # looking at the distribution chart we see two peaks and two troughs.
    # Most likely, the two peaks are transactions during the day because of the high volume of transactions,
    # and the two bottoms are transactions at night when everyone is asleep.

    fig, axs = plt.subplots(ncols=2, figsize=(16, 4))
    axs[0].set_title("Distribution of Fraud Transactions")
    sns.histplot(fraud["time"], bins=100, color="red", ax=axs[0])

    sns.histplot(normal["time"], bins=100, color="cyan", ax=axs[1])
    axs[1].set_title("Distribution of Genuine Transactions")

    if save_graphs:
        plt.savefig("images/dataset/time_distribution.png", transparent=True)

    fig, axs = plt.subplots(ncols=2, figsize=(16, 4))
    sns.histplot(fraud["amount"], bins=100, color="red", ax=axs[0])
    axs[0].set_title("Distribution of Fraud Transactions")

    sns.histplot(normal["amount"], bins=100, color="cyan", ax=axs[1])
    axs[1].set_title("Distribution of Normal Transactions")

    if save_graphs:
        plt.savefig("images/dataset/transaction_distribution.png", transparent=True)

    # Log transforms are useful when applied to skewed distributions
    # because they tend to expand values in the lower magnitude range
    # and tend to compress or reduce values in the magnitude range.

    # Scale amount by log

    df.insert_column(-1, pd.Series("amount_log", np.log(df["amount"] + 0.0001)))

    print(df.head())

    # Convert everything to float, otherwise we get errors with string - float operations
    _tmp = df.get_column("class").cast(pd.Float64)
    df = df.drop("class")
    df.insert_column(-1, _tmp)

    # Feature correlation study
    corr = df.corr().to_pandas()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, (ax1) = plt.subplots(1, 1, figsize=(18, 8))
    sns.heatmap(
        df.corr().to_pandas(), vmax=0.8, square=True, ax=ax1, cmap="magma_r", mask=mask
    )
    ax1.set_title("Whole Dataset feature correlation")
    ax1.set_yticklabels(df.columns, rotation=0)
    if save_graphs:
        plt.savefig("images/dataset/feature_correlation.png", transparent=True)
    if show_graphs:
        plt.show()

    # Prepare plot in which we'll put two correlation matrix (fraud and normal)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Get data about fraud transaction vs normal transactions (count, columns)
    fraud = df.filter(pd.col("class") == 1)
    normal = df.filter(pd.col("class") == 0)

    # mask used to remove useless (mirrored) data.
    sns.heatmap(
        fraud.corr().to_pandas(),
        vmax=0.8,
        square=True,
        ax=ax1,
        cmap="rocket_r",
        mask=mask,
    )
    ax1.set_title("Fraud")
    ax1.set_yticklabels(df.columns, rotation=0)
    sns.heatmap(
        normal.corr().to_pandas(),
        vmax=0.8,
        square=True,
        ax=ax2,
        cmap="YlGnBu",
        mask=mask,
    )
    ax2.set_title("Normal")
    ax2.set_yticklabels(df.columns, rotation=0)
    if save_graphs:
        plt.savefig(
            "images/dataset/feature_correlation_comparison.png", transparent=True
        )
    if show_graphs:
        plt.show()

    # Time inverse correlation
    plot_feature_corr_scatter(
        df, ("v3", "time"), "time_and_v3_inv_correlation", save_graphs
    )

    # Amount correlation

    # Direct correlation
    plot_feature_corr_scatter(
        df, ("v20", "amount"), "amount_and_v20_correlation", save_graphs
    )
    plot_feature_corr_scatter(
        df, ("v7", "amount"), "amount_and_v7_correlation", save_graphs
    )

    # Inverse correlation
    plot_feature_corr_scatter(
        df, ("v1", "amount"), "amount_and_v1_inv_correlation", save_graphs
    )
    plot_feature_corr_scatter(
        df, ("v2", "amount"), "amount_and_v2_inv_correlation", save_graphs
    )
    plot_feature_corr_scatter(
        df, ("v5", "amount"), "amount_and_v5_inv_correlation", save_graphs
    )

    if show_graphs:
        plt.show()
    return df


def plot_feature_corr_scatter(
    df: DataFrame, features: (str, str), file_name: str, save_graph: bool = False
):
    first_key, second_key = features
    s = sns.lmplot(
        x=first_key,
        y=second_key,
        data=df,
        hue="class",
        fit_reg=True,
        scatter_kws={"s": 2},
    )
    r, _ = pearsonr(df.get_column(first_key), df.get_column(second_key))
    plt.text(
        0.95,
        0.85,
        f"r = {r:.2f}",
        ha="right",
        va="center",
        transform=s.ax.transAxes,
    )
    if save_graph:
        plt.savefig(f"images/dataset/{file_name}.png", transparent=True)
