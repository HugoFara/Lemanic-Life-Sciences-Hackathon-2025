import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import ast

def compute_kappa(csv_path):
    df = pd.read_csv(csv_path)
    all_model, all_coder1, all_coder2 = [], [], []

    for idx, row in df.iterrows():
        try:
            model_preds = ast.literal_eval(row['outcome'])  # ex: [1, 0, 1, 1]
            coder1 = list(map(int, str(row['accuracy_coder1']).split()))  # ex: "1 0 1"
            coder2 = list(map(int, str(row['accuracy_coder2']).split()))
        except Exception as e:
            print(f"[⚠️] Ligne {idx} ignorée (erreur parsing) : {e}")
            continue

        if not (len(model_preds) == len(coder1) == len(coder2)):
            print(f"[⚠️] Ligne {idx} ignorée (longueurs différentes)")
            continue

        all_model.extend(model_preds)
        all_coder1.extend(coder1)
        all_coder2.extend(coder2)

    if not all_model:
        return "Aucune donnée valide."

    kappa1 = cohen_kappa_score(all_model, all_coder1)
    kappa2 = cohen_kappa_score(all_model, all_coder2)
    kappa3 = cohen_kappa_score(all_coder1, all_coder2)

    counts = np.zeros((len(all_model), 2), dtype=int)
    for i in range(len(all_model)):
        for vote in [all_model[i], all_coder1[i], all_coder2[i]]:
            counts[i, int(vote)] += 1

    fleiss = fleiss_kappa(counts)

    return {
        'kappa_model_vs_coder1': kappa1,
        'kappa_model_vs_coder2': kappa2,
        'kappa_coder1_vs_coder2': kappa3,
        'fleiss_kappa': fleiss
    }