import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Aggiungi la root del progetto al path per gli import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Python.XAI.plot_ndcg import XAIPlotter
from Python.utils.py_logger import CustomLogger

class TestXAIPlotter(unittest.TestCase):
    def setUp(self):
        self.logger = CustomLogger(name="TestPlotter")
        self.plotter = XAIPlotter(logger=self.logger)

    @patch('Python.XAI.plot_ndcg.plt.savefig')
    @patch('Python.XAI.plot_ndcg.os.makedirs')
    def test_plot_top_rois(self, mock_makedirs, mock_savefig):
        """Testa se la funzione di plotting genera correttamente il grafico senza errori."""
                 
        # Crea un DataFrame fittizio simile a quello restituito da ROIAnalyzer
        data = {
            'ROI_ID': [1, 2, 3],
            'ROI_Name': ['Region A', 'Region B', 'Region C'],
            'Mean_ROI_Signal': [0.5, 0.8, 0.2],
            'Sum_ROI_Signal': [50.0, 80.0, 20.0]
        }
        df = pd.DataFrame(data)

        # Testa il plot usando la metrica Mean
        self.plotter.plot_top_rois(df, score_col='Mean_ROI_Signal', title='Test Plot', top_k=2, out_path="fake_path.png")
                 
        # Verifica che il percorso sia stato creato
        mock_makedirs.assert_called_once()
        # Verifica che il file sia stato salvato
        mock_savefig.assert_called_once()

if __name__ == '__main__':
    unittest.main()