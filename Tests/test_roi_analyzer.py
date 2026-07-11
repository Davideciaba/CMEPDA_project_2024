import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import sys
import os

# Costruisce il percorso assoluto alla cartella "Python" del tuo progetto
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
python_dir = os.path.join(project_root, 'Python')

# Aggiunge la cartella "Python" al PYTHONPATH
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

# Ora gli import relativi a "Python" funzioneranno
from XAI.roi_analyzer import ROIAnalyzer
from utils.py_logger import CustomLogger

class TestROIAnalyzer(unittest.TestCase):

    def setUp(self):
        self.logger = CustomLogger(name="TestROI")
        self.analyzer = ROIAnalyzer(logger=self.logger)

    @patch('XAI.roi_analyzer.os.path.exists')
    @patch('XAI.roi_analyzer.nib.load')
    @patch('XAI.roi_analyzer.pd.read_csv')
    def test_extract_regional_importance_with_filtering(self, mock_read_csv, mock_nib_load, mock_exists):
        """Testa l'estrazione delle ROI e il filtro della Materia Bianca/Ventricoli."""
        mock_exists.return_value = True
        
        # Simula un CSV con regioni miste (GM, WM, Ventricoli)
        mock_df = pd.DataFrame({
            'ROI_ID': [1, 2, 3, 4],
            'ROI_Name': ['Left Hippocampus', 'Right Cerebral White Matter', 'Left Lateral Ventricle', 'Right Amygdala']
        })
        mock_read_csv.return_value = mock_df
        
        # Simula un atlante (4 regioni, ciascuna con 2 voxel per semplicità)
        atlas_vol = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4]
        ])
        mock_atlas = MagicMock()
        mock_atlas.get_fdata.return_value = atlas_vol
        
        # Simula una mappa XAI con valori arbitrari
        xai_vol = np.array([
            [0.5, 0.5], # Hippocampus (Mean = 0.5)
            [1.0, 1.0], # White Matter (Dovrebbe essere filtrata)
            [2.0, 2.0], # Ventricle (Dovrebbe essere filtrato)
            [0.8, 0.8]  # Amygdala (Mean = 0.8)
        ])
        mock_xai = MagicMock()
        mock_xai.get_fdata.return_value = xai_vol
        
        mock_nib_load.side_effect = [mock_xai, mock_atlas]
        
        # Esecuzione
        df_result = self.analyzer.extract_regional_importance('fake_xai.nii', 'fake_atlas.nii', 'fake_csv.csv')
        
        # Asserzioni
        # 1. Devono rimanere solo 2 regioni (Ippocampo e Amigdala)
        self.assertEqual(len(df_result), 2)
        
        # 2. Le regioni filtrate non devono esserci
        region_names = df_result['ROI_Name'].tolist()
        self.assertNotIn('Right Cerebral White Matter', region_names)
        self.assertNotIn('Left Lateral Ventricle', region_names)
        
        # 3. L'Amigdala (0.8) deve essere prima dell'Ippocampo (0.5) perché i risultati sono ordinati in modo decrescente
        self.assertEqual(df_result.iloc[0]['ROI_Name'], 'Right Amygdala')
        self.assertEqual(df_result.iloc[1]['ROI_Name'], 'Left Hippocampus')

    def test_ndcg_calculation(self):
        """Testa la correttezza matematica del calcolo dell'nDCG."""
        # Scenario di test standard
        # Valori veri di rilevanza (ground truth)
        true_scores = np.array([3, 2, 3, 0, 1, 2])
        # Valori predetti (le previsioni del modello per quegli stessi elementi)
        predicted_scores = np.array([2, 1, 3, 0, 0, 1]) 
        
        # Calcolo manuale per k=3:
        # 1. Ordine ideale (basato su true_scores): [3, 3, 2, 2, 1, 0]
        #    Top 3 ideali: [3, 3, 2]
        #    IDCG@3 = 3/log2(2) + 3/log2(3) + 2/log2(4) = 3 + 1.892 + 1 = 5.892
        #
        # 2. Ordine predetto (indici ordinati per predicted_scores): [2, 0, 1, 5, 4, 3]
        #    Valori *veri* corrispondenti all'ordine predetto: [3, 3, 2, 2, 1, 0]
        #    (In questo caso specifico, l'ordine predetto produce gli stessi primi 3 elementi dell'ordine ideale)
        #    DCG@3 = 3/log2(2) + 3/log2(3) + 2/log2(4) = 5.892
        #
        # 3. nDCG = DCG / IDCG = 1.0
        
        ndcg_k3 = self.analyzer.calculate_ndcg(predicted_scores, true_scores, k=3)
        self.assertAlmostEqual(ndcg_k3, 1.0, places=3)
        
        # Facciamo un test in cui l'ordine predetto è pessimo
        bad_predictions = np.array([0, 0, 0, 3, 2, 1])
        # Ordine predetto: [3, 4, 5, 0, 1, 2]
        # Valori *veri* corrispondenti: [0, 1, 2, 3, 2, 3]
        # Top 3 presi: [0, 1, 2]
        # DCG@3 = 0/log2(2) + 1/log2(3) + 2/log2(4) = 0 + 0.6309 + 1 = 1.6309
        # nDCG = 1.6309 / 5.892 = 0.2768
        
        ndcg_bad_k3 = self.analyzer.calculate_ndcg(bad_predictions, true_scores, k=3)
        self.assertTrue(ndcg_bad_k3 < 1.0)
        self.assertAlmostEqual(ndcg_bad_k3, 0.2768, places=3)

if __name__ == '__main__':
    unittest.main()