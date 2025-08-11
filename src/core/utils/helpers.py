from multiprocessing import Condition, Event, Queue
from multiprocessing.managers import BaseManager
from typing import TypeVar, Generic

T = TypeVar("T")


class MultiprocessingDequeue(Generic[T]):
    def __init__(self, queue: "Queue[T]") -> None:
        self.queue = queue
        self.condition = Condition()
        self.event = Event()

    def append(self, item: T):
        with self.condition:
            if self.queue.full():
                self.queue.get()
            self.queue.put(item)
            self.condition.notify_all()
        self.event.set()

    def wait_for_data(self):
        self.event.wait()
        self.event.clear()

    def popleft(self) -> T | None:
        with self.condition:
            if not self.queue.empty():
                return self.queue.get()
        return None

    def popleft_blocking(self) -> T:
        with self.condition:
            self.condition.wait(5)
            return self.queue.get()


class MySharedDictManager(BaseManager):
    pass

_shared_dict_attr_name = 'shared_dict'
MySharedDictManager.register(_shared_dict_attr_name, dict)
manager = MySharedDictManager()
manager.start()


class SharedMemoryObject:
    """
    A base class that uses a shared dictionary from a multiprocessing.Manager
    to create a dataclass-like object for shared state.
    """
    def __init__(self, **kwargs):
        # Store the shared dictionary in a "private" attribute to avoid
        # infinite loops with __setattr__.
        assert hasattr(manager, _shared_dict_attr_name)
        get_shared_dict = getattr(manager, _shared_dict_attr_name)
        super().__setattr__('_shared_dict', get_shared_dict())

        # Get the defined fields from the class's type annotations.
        # This is how we know what fields this "dataclass" has.
        fields = self.__class__.__annotations__

        # Initialize the shared dictionary with provided keyword arguments
        # or set to None if not provided.
        for field in fields:
            self._shared_dict[field] = kwargs.get(field)

    def __getattr__(self, name):
        """
        Called when you try to access an attribute (e.g., settings.timeout).
        """
        if name in self._shared_dict:
            return self._shared_dict[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Called when you try to set an attribute (e.g., settings.timeout = 60).
        """
        # We must check if the attribute is in the annotations to prevent
        # creating new, non-shared attributes accidentally.
        if name not in self.__class__.__annotations__:
            raise AttributeError(f"Cannot set new attribute '{name}'. Only defined fields can be set.")

        # Update the value in the shared dictionary.
        self._shared_dict[name] = value

    def __repr__(self):
        """
        Provides a nice, readable representation of the object's current state.
        """
        items = self._shared_dict.items()
        item_strs = [f"{key}={repr(value)}" for key, value in items]
        return f"{self.__class__.__name__}({', '.join(item_strs)})"