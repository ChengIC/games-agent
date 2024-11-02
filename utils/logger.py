import logging
from datetime import datetime
import os

class ExperimentLogger:
    def __init__(self, log_dir='logs', game_id=None):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.game_id = game_id
        
        # Create timestamp-based subfolder
        self.timestamp_dir = os.path.join(log_dir, self.timestamp)
        if not os.path.exists(self.timestamp_dir):
            os.makedirs(self.timestamp_dir)
        
        self.log_file = self._create_log_file()
        
        # Configure the logger
        self.logger = logging.getLogger(f"Experiment_{self.timestamp}_{self.game_id}")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add only the file handler to logger
        self.logger.addHandler(file_handler)

    def _create_log_file(self):
        if self.game_id:
            return os.path.join(self.timestamp_dir, f"game_{self.game_id}.log")
        return os.path.join(self.timestamp_dir, "experiment.log")

    def log(self, message, level='info'):
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)

    def validate_updated_nodes(self, updated_nodes):
        """
        Validate that the updated nodes follow the state pattern:

        [host, call_tool, host, player, call_tool, player, ...]

        """
        pattern = ["host", "call_tool", "host", "player", "call_tool", "player"]
        for idx, node in enumerate(updated_nodes):
            expected_node = pattern[idx % len(pattern)]
            if node != expected_node:
                self.log(f"Invalid node order: {node} at index {idx} for updated_nodes: {updated_nodes}", level="error")
        
        self.log(f"Updated nodes are valid: {updated_nodes}", level="info")

# Remove the global instance
# experiment_logger = ExperimentLogger()