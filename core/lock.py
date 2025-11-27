from readerwriterlock import rwlock
import functools

class GlobalLock:
    _instance = None
    _lock = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalLock, cls).__new__(cls)
            # Priority Writer Lock to prevent writer starvation
            cls._lock = rwlock.RWLockWrite()
        return cls._instance

    @property
    def lock(self):
        return self._lock

# Singleton instance
global_lock = GlobalLock().lock

def read_locked(func):
    """Decorator for read-heavy operations (concurrent access)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with global_lock.gen_rlock():
            return func(*args, **kwargs)
    return wrapper

def write_locked(func):
    """Decorator for write operations (exclusive access)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with global_lock.gen_wlock():
            return func(*args, **kwargs)
    return wrapper
