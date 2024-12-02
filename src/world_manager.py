import threading
import esper

class WorldManager:
    def __init__(self):
        self._world = esper.World()
        self._lock = threading.Lock()
    
    def with_world(self):
        return self._lock
        
    @property
    def world(self):
        return self._world

world_manager = WorldManager() 