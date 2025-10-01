import logging

class Logger:
    def __init__(self,name:str,config):
        self.config = config.get('logger',{})
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        
        # 尝试从不同位置获取log_dir
        log_dir = self.config.get('log_dir') or config.get('Data', {}).get('log_dir') or config.get('log_dir')
        if log_dir:
            import os
            os.makedirs(log_dir, exist_ok=True)
            self.logger.addHandler(logging.FileHandler(os.path.join(log_dir, name+'.log')))

    def get_logger(self):
        return self.logger
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def critical(self, message):
        self.logger.critical(message)
    
    def exception(self, message):
        self.logger.exception(message)

    