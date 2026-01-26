import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path("ANLP_project_refactored\EVAL_folder")

criteria = ["Relevance", "Severity", "Humor", "Concreteness"]
times = [
    "Syn- antonym gen time",
    "Syn- antonym cleaning time",
    "Comparator gen time",
    "Comparator choose time",
]

def load_all():
    eval_rows = []
    fav_rows = []

    for person_dir in BASE_DIR.iterdir(): # doe eval per persoon
        if not person_dir.is_dir():
            continue
        person = person_dir.name
        for csv_file in person_dir.glob("*.csv"): # eval per methode
            m = re.match(
                r"evaluation_log\d+_(.+?)_(.+?)_(.+?)\.csv",
                csv_file.name,
            )
            if not m:
                print(f"Skipping unexpected filename: {csv_file.name}")
                continue

            vec_model, sys_ant_model, projection_model = m.groups()

            df = pd.read_csv(csv_file)
            df["person"] = person
            df["vec_model"] = vec_model
            df["sys_ant_model"] = sys_ant_model
            df["projection_model"] = projection_model

            df["Concreteness"] = (
                df["Concreteness"]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({"Y": 1, "N": 0})
            )
            df["Concreteness"] = pd.to_numeric(df["Concreteness"], errors="coerce")


            is_favorite = (df["sys_ant_model"].isna() | (df["sys_ant_model"] == "") | df["Relevance"].isna())

            df_eval = df[~is_favorite].copy() 
            df_fav = df[is_favorite].copy()

            df_eval[criteria + times] = df_eval[criteria + times].apply(pd.to_numeric, errors="coerce")

            eval_rows.append(df_eval)
            fav_rows.append(df_fav)

    df_eval_all = pd.concat(eval_rows, ignore_index=True)
    df_fav_all = pd.concat(fav_rows, ignore_index=True)

    return df_eval_all, df_fav_all

def add_insult_id(df): # Creeer uniek ID per geevalueerd woord
    return (
        df.assign(
            insult_id=lambda x: (
                x["person"] + "_" +
                x["insult"] + "_" +
                x["vec_model"] + "_" +
                x["sys_ant_model"] + "_" +
                x["projection_model"]
            )
        )
    )

def winner_candidates(df_eval, df_fav): 
    winners = (
        df_eval
        .merge(
            df_fav[["insult_id", "word"]],
            on="insult_id",
            how="inner"
        )
        .query("word_x == word_y")
        .groupby("insult_id")[criteria]
        .mean()
        .rename(columns=lambda x: f"{x}_winner")
    )

    return winners

def winner_percentile(df_eval, df_fav):
    rows = []

    for iid, group in df_eval.groupby("insult_id"):
        winner = df_fav.loc[df_fav["insult_id"] == iid, "word"].values
        if len(winner) == 0:
            continue

        winner = winner[0]

        for metric in criteria:
            scores = group[metric].dropna()
            winner_score = group.loc[group["word"] == winner, metric]

            if winner_score.empty:
                continue

            percentile = (scores < winner_score.iloc[0]).mean()
            rows.append({
                "insult_id": iid,
                "metric": metric,
                "percentile": percentile
            })

    return pd.DataFrame(rows)

def method_quality(df_eval):
    df = df_eval.copy()

    # Compute total runtime per row
    df["total_time"] = (
        df["Syn- antonym gen time"] +
        df["Syn- antonym cleaning time"] +
        df["Comparator gen time"] +
        df["Comparator choose time"]
    )

    return (
        df
        .groupby(["vec_model", "sys_ant_model", "projection_model"])
        .agg(
            Relevance=("Relevance", "mean"),
            Severity=("Severity", "mean"),
            Humor=("Humor", "mean"),
            Concreteness=("Concreteness", "mean"),
            total_time=("total_time", "mean"),
        )
        .sort_values("Relevance", ascending=False)
    )


def top_selected_mean(df_eval):
    """
    Compute the mean Relevance, Severity, Humor of the **first candidate** per insult.
    """
    # Group by insult_id and take the first row
    top_rows = df_eval.groupby("insult_id").first()

    # Compute mean across all insults
    top_mean = top_rows[criteria].mean()
    top_mean.name = "Top_candidate_mean"
    return top_mean

def annotator_bias(df_eval):
    return (
        df_eval
        .groupby("person")[criteria]
        .mean()
    )

df_eval, df_fav = load_all()
df_eval = add_insult_id(df_eval)
df_fav = add_insult_id(df_fav)

# Candidate average (all 5 rows)
cand_mean = df_eval.groupby("insult_id")[criteria].mean().mean()
print("Candidate mean (all candidates per insult):\n", cand_mean)

# Top selected (by model itself)
top_mean = top_selected_mean(df_eval)
print("\nTop candidate mean (first candidate per insult):\n", top_mean)

# Winner (by human)
winner_mean = winner_candidates(df_eval, df_fav).mean()
print("\nFavorite candidate mean (favorite candidate per insult):\n", winner_mean)

print("\nFavorite Percentiles (How much better is favorites compared to the rest):")
df_percentiles = winner_percentile(df_eval, df_fav)
percentile_means = df_percentiles.groupby("metric")["percentile"].mean()
print(percentile_means)

print("\nMethod performance:")
print(method_quality(df_eval))

print("\nAnnotator Bias:")
print(annotator_bias(df_eval))




