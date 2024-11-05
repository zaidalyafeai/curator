# Abstract class for a postgres database. In case we have the logic for _store_metadata in bella.py here. 
from abc import ABC, abstractmethod

class Database(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def store_metadata(self, metadata: dict):
        pass

