import threading

class WindowManager:
    def __init__(self):
        self._window = None
        self._lock = threading.Lock()
    
    def with_window(self):
        return self._lock
    
    @property
    def window(self):
        return self._window
    
    @window.setter
    def window(self, value):
        self._window = value

window_manager = WindowManager() 