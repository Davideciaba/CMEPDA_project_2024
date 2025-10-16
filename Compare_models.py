# %% STEP 9: COMPARE SVM AND EFFICIENTNET3D MODELS
# Author: Gemini
# Date: 16/10/2025
# --------------------------------------------------------------------------
# Questo script esegue il confronto finale tra i risultati della pipeline SVM
# e della pipeline EfficientNet3D a livello di "aspetti" (cluster).
# 1. Carica i vettori di importanza finale (W_bar_hold e Delta_G_hold).
# 2. Calcola la correlazione di Spearman tra i due modelli.
# 3. Classifica gli aspetti più importanti secondo ciascun modello.
# 4. Crea un grafico a barre comparativo per la visualizzazione.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from loguru import logger
import sys

# --- 0. Configurazione del Logger ---
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# --- 1. Definizione dei Percorsi dei File di Input ---
svm_results_file = 'svm_results.npz'
efficientnet_results_file = 'efficientnet_results.npz'
clustering_file = 'hierarchical_clustering_results.npz'

# --- 2. Caricamento dei Vettori di Importanza ---
logger.info("Caricamento dei risultati finali dalle pipeline SVM e EfficientNet...")
try:
    with np.load(svm_results_file) as data:
        W_bar_hold = data['W_bar_hold']
    
    with np.load(efficientnet_results_file) as data:
        Delta_G_hold = data['Delta_G_hold']
        
    with np.load(clustering_file) as data:
        K = int(data['num_clusters'])

    if len(W_bar_hold) != K or len(Delta_G_hold) != K:
        raise ValueError("La dimensione dei vettori di importanza non corrisponde al numero di cluster.")
        
    logger.success(f"Dati caricati con successo per K = {K} aspetti.")

except (FileNotFoundError, ValueError) as e:
    logger.critical(f"Errore nel caricamento dei file: {e}. Assicurati di aver eseguito prima 'SVM.py' e 'EfficientNet.py'.")
    sys.exit(1)

# --- 3. Creazione di un DataFrame per l'Analisi ---
# Usare un DataFrame di Pandas semplifica l'analisi e la visualizzazione.
df = pd.DataFrame({
    'Aspect': np.arange(1, K + 1),
    'SVM_Score': W_bar_hold,
    'EfficientNet_Score': Delta_G_hold
})

# --- 4. Calcolo della Correlazione tra i Modelli ---
# Usiamo la correlazione di Spearman come descritto nella pipeline,
# che è robusta a differenze di scala e a relazioni non lineari.
correlation, p_value = spearmanr(df['SVM_Score'], df['EfficientNet_Score'])
logger.success(f"Correlazione di Spearman tra i vettori di importanza: rho = {correlation:.4f} (p-value = {p_value:.4f})")
if p_value < 0.05:
    logger.info("La correlazione è statisticamente significativa, indicando un buon accordo tra i modelli.")
else:
    logger.warning("La correlazione non è statisticamente significativa.")

# --- 5. Ranking degli Aspetti più Importanti ---
logger.info("Classifica dei 5 aspetti più importanti per ciascun modello (basata sul valore assoluto):")

# Aggiungiamo i valori assoluti per il ranking
df['SVM_Abs_Score'] = df['SVM_Score'].abs()
df['EffNet_Abs_Score'] = df['EfficientNet_Score'].abs()

top5_svm = df.sort_values(by='SVM_Abs_Score', ascending=False).head(5)
top5_effnet = df.sort_values(by='EffNet_Abs_Score', ascending=False).head(5)

print("\n--- Top 5 Aspetti per SVM ---")
print(top5_svm[['Aspect', 'SVM_Score']].to_string(index=False))

print("\n--- Top 5 Aspetti per EfficientNet ---")
print(top5_effnet[['Aspect', 'EfficientNet_Score']].to_string(index=False))

# --- 6. Visualizzazione Comparativa ---
logger.info("Creazione del grafico a barre comparativo...")

# Normalizziamo i punteggi (z-score) per renderli visivamente confrontabili,
# poiché le loro scale native (pesi SVM vs. attivazioni GradCAM) sono diverse.
df['SVM_Score_Z'] = (df['SVM_Score'] - df['SVM_Score'].mean()) / df['SVM_Score'].std()
df['EfficientNet_Score_Z'] = (df['EfficientNet_Score'] - df['EfficientNet_Score'].mean()) / df['EfficientNet_Score'].std()

# Trasformiamo il DataFrame per la visualizzazione con seaborn
df_melted = df.melt(id_vars='Aspect', 
                    value_vars=['SVM_Score_Z', 'EfficientNet_Score_Z'],
                    var_name='Model', 
                    value_name='Normalized Importance Score')
df_melted['Model'] = df_melted['Model'].replace({'SVM_Score_Z': 'SVM', 'EfficientNet_Score_Z': 'EfficientNet'})

# Creazione del grafico
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(data=df_melted, x='Aspect', y='Normalized Importance Score', hue='Model', ax=ax)

ax.set_title('Confronto dell\'Importanza degli Aspetti tra SVM e EfficientNet', fontsize=16, weight='bold')
ax.set_xlabel('Aspetto (Cluster)', fontsize=12)
ax.set_ylabel('Importanza Normalizzata (Z-score)', fontsize=12)
ax.axhline(0, color='black', linewidth=0.8) # Linea dello zero
ax.legend(title='Modello')
plt.tight_layout()

output_plot_file = 'model_comparison_plot.png'
fig.savefig(output_plot_file, dpi=300)
logger.success(f"Grafico di confronto salvato in '{output_plot_file}'")