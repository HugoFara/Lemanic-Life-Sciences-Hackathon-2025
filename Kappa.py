import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import os

def compute_kappa(audio_path, csv_path, binary_results):
    import ast
    import re

    # Extraire le nom de fichier
    audio_name = os.path.basename(audio_path)

    # Charger le CSV
    df = pd.read_csv(csv_path)

    # Filtrer la ligne correspondant à l'audio
    row = df[df['file_name'] == audio_name]
    if row.empty:
        return f"Audio '{audio_name}' non trouvé dans le CSV."

    # Convertir les chaînes d'entiers séparés par des espaces en listes de 0/1
    parse_list = lambda s: list(map(int, re.findall(r'\d+', str(s))))
    labels1 = parse_list(row['accuracy_coder1'].values[0])
    labels2 = parse_list(row['accuracy_coder2'].values[0])

    # Vérifier la longueur
    if not (len(binary_results) == len(labels1) == len(labels2)):
        return "Erreur : les longueurs des listes ne correspondent pas."

    # Calcul des kappa de Cohen
    kappa1 = cohen_kappa_score(binary_results, labels1)
    kappa2 = cohen_kappa_score(binary_results, labels2)

    # Préparer les données pour Fleiss' Kappa
    counts = np.zeros((len(binary_results), 2), dtype=int)
    for i in range(len(binary_results)):
        for vote in [labels1[i], labels2[i], binary_results[i]]:
            counts[i, int(vote)] += 1

    fleiss = fleiss_kappa(counts)

    return {
        'kappa_model_vs_coder1': kappa1,
        'kappa_model_vs_coder2': kappa2,
        'fleiss_kappa': fleiss
    }

"""
Example usage:
resultats = kappa.compute_kappa("Hackathon_ASR/2_Audiofiles/Decoding_IT_T1/102_edugame2023_32c4a5e851c1431aba3aa409e3be8128_649d404f44214261b67b24f1845e1350.wav", "Hackathon_ASR/1_Ground_truth/Decoding_ground_truth_IT.csv", binary_results=[1 ,0 ,1 ,0 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1])
print(resultats)
"""