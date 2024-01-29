# from queue import SimpleQueue
# class StdOut:
#     def __init__(self) -> None:
#         self._queue = SimpleQueue()
    
#     def write(self, string: str):
#         self._queue.put_nowait(string)
    
#     def read(self):
#         if not self.empty():
#             return self._queue.get_nowait()
        
    
#     def empty(self):
#         return self._queue.empty()
    
#     def close(self):
#         pass

import os
class StdOut:
    def __init__(self) -> None:
        _r, _w = os.pipe()
        self._istream, self._ostream = os.fdopen(_r, "r"), os.fdopen(_w, "w", 1)
    
    def write(self, string: str):
        try:
            self._ostream.write(string + "\n")
        except:
            pass
    
    def read(self):
        try:
            return self._istream.readline()
        except:
            return None
        
    
    def empty(self):
        return self._istream.closed
    
    def close(self):
        self._ostream.close()
        self._istream.close()