import sys
from loguru import logger

class CustomLogger:
    """Classe per la gestione del logging personalizzato.
    Utilizza la libreria loguru.
    """
    def __init__(self, log_file_path='preliminaries.log', enable_file_logging=False):
        """
        Inizializza il logger.

        Args:
            log_file_path (str): Il percorso del file di log.
            enable_file_logging (bool): Se True, attiva la scrittura su file.
                                        Di default è False.
        """
        # 1. Rimuoviamo il logger di default per avere pieno controllo
        logger.remove()

        # 2. Aggiungiamo un output per la CONSOLE
        # Formato personalizzato per essere leggibile e colorato
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[session_id]}</cyan> - <level>{message}</level>",
            colorize=True
        )
        # 3. Aggiungiamo un output per il FILE
        if enable_file_logging:
            logger.add(
                log_file_path,
                level="DEBUG",
                rotation="50 KB",
                compression="zip",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[session_id]} - {message}"
            )
            logger.info(f"File logging ABILITATO. Scrittura su '{log_file_path}'")
        else:
            logger.info("File logging DISABILITATO. Solo output su console.")

    def log(self, level, message):
        """Delega la chiamata a Loguru."""
        logger.log(level.upper(), message)

    def context(self, **kwargs):
        """Restituisce un contesto con informazioni extra per il logger."""
        return logger.contextualize(**kwargs)