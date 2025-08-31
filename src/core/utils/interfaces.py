import atexit
import dataclasses
import inspect
import logging
from ctypes import c_int
from multiprocessing import Condition, Event, Lock, Manager, Queue, Value
from multiprocessing.shared_memory import SharedMemory
from typing import Generic, TypeVar

import numpy as np

# from .shared import T, manager

logger = logging.getLogger(__name__)

T = TypeVar("T")
manager = Manager()
atexit.register(manager.shutdown)


def is_shared_memory_subclass(cls):
    """Helper to safely check if a type is a subclass of SharedMemoryObject."""
    return inspect.isclass(cls) and issubclass(cls, SharedMemoryObject)


@dataclasses.dataclass
class ForegroundMaskOptions:
    mog2_history = Value(c_int, 500)
    mog2_var_threshold = Value(c_int, 16)
    denoise_kernelsize = Value(c_int, 7)


class TuningSettings:
    def __init__(self):
        self.foreground_mask_options = ForegroundMaskOptions()

    def update(self):
        pass


class DoubleBuffer:
    """
    A synchronized, shared-memory double buffer for efficient, non-blocking
    frame passing between a high-speed producer and a slower consumer.
    """

    def __init__(self, name: str, shape: tuple, dtype: np.dtype):
        """
        Initializes the two shared memory buffers.
        Args:
            name: A unique name prefix for the shared memory blocks.
            shape: The numpy shape of the data (e.g., (1080, 1920, 3)).
            dtype: The numpy data type (e.g., np.uint8).
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
        size_in_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

        try:
            self._shm_a = SharedMemory(create=True, size=size_in_bytes, name=f"shm_{name}_a")
            self._shm_b = SharedMemory(create=True, size=size_in_bytes, name=f"shm_{name}_b")
        except FileExistsError:
            logger.warning(f"Shared memory for '{name}' already exists. Unlinking and recreating.")
            SharedMemory(name=f"shm_{name}_a").unlink()
            SharedMemory(name=f"shm_{name}_b").unlink()
            self._shm_a = SharedMemory(create=True, size=size_in_bytes, name=f"shm_{name}_a")
            self._shm_b = SharedMemory(create=True, size=size_in_bytes, name=f"shm_{name}_b")

        self._buffer_a = np.ndarray(shape, dtype=dtype, buffer=self._shm_a.buf)
        self._buffer_b = np.ndarray(shape, dtype=dtype, buffer=self._shm_b.buf)
        self._buffers = [self._buffer_a, self._buffer_b]
        self._lock = Lock()
        self._write_index = 0

    def write(self, frame: np.ndarray):
        """Writes a new frame to the current back buffer."""
        self._buffers[self._write_index][:] = frame

    def read_and_swap(self) -> np.ndarray:
        """Atomically swaps buffers and returns a copy of the new front buffer."""
        with self._lock:
            read_idx = self._write_index
            self._write_index = 1 - read_idx
        return self._buffers[read_idx].copy()

    def close(self):
        """Closes and unlinks the shared memory blocks."""
        logger.info(f"Closing and unlinking double buffer '{self.name}'...")
        self._shm_a.close()
        self._shm_a.unlink()
        self._shm_b.close()
        self._shm_b.unlink()


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


class SharedMemoryObject:
    """
    A base class that acts like a dataclass but uses a shared
    dictionary for its state, supporting nested instances.
    """

    def __init__(self, _dict_proxy=None, **kwargs):
        # Initialize the shared dictionary for this instance
        shared_dict = _dict_proxy if _dict_proxy is not None else manager.dict()
        super().__setattr__("_shared_dict", shared_dict)

        all_fields = self.__class__.__annotations__

        # Handle all provided keyword arguments
        for field_name, provided_value in kwargs.items():
            if field_name not in all_fields:
                raise TypeError(f"__init__() got an unexpected keyword argument '{field_name}'")

            self._initialize_field(field_name, all_fields[field_name], provided_value)

        # Handle any fields that were NOT provided and need default initialization
        for field_name, field_type in all_fields.items():
            if field_name not in kwargs:
                self._initialize_field(field_name, field_type)

    def _initialize_field(self, name, type_hint, value=None):
        """Helper method to initialize a single field."""
        is_nested_type = is_shared_memory_subclass(type_hint)

        if is_nested_type:
            # Case 1: An instance was passed (e.g., debug_settings=DebugSettings())
            if isinstance(value, SharedMemoryObject):
                nested_obj = value
                # Adopt the existing object's shared dictionary
                # noinspection PyProtectedMember
                self._shared_dict[name] = nested_obj._shared_dict
                super().__setattr__(name, nested_obj)

            # Case 2: A dictionary was passed (e.g., debug_settings={'render_bboxes': True})
            elif isinstance(value, dict):
                nested_dict = manager.dict()
                self._shared_dict[name] = nested_dict
                nested_obj = type_hint(_dict_proxy=nested_dict, **value)
                super().__setattr__(name, nested_obj)

            # Case 3: Nothing was passed, so create a default empty instance
            elif value is None:
                nested_dict = manager.dict()
                self._shared_dict[name] = nested_dict
                nested_obj = type_hint(_dict_proxy=nested_dict)
                super().__setattr__(name, nested_obj)
            else:
                raise TypeError(
                    f"Argument '{name}' must be a dict or a {type_hint.__name__} instance, not {type(value).__name__}"
                )
        else:
            # It's a primitive type, just assign the value (or None if not provided)
            self._shared_dict[name] = value

    def __getattr__(self, name):
        """Called when accessing an attribute that isn't found normally."""
        # We need to handle the case where we are accessing a nested object instance
        if name in self.__class__.__annotations__ and is_shared_memory_subclass(self.__class__.__annotations__[name]):
            return super().__getattribute__(name)

        if name in self._shared_dict:
            return self._shared_dict[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Called when setting any attribute."""
        if name not in self.__class__.__annotations__:
            raise AttributeError(f"Cannot set new attribute '{name}'. Only defined fields can be set.")

        field_type = self.__class__.__annotations__.get(name)
        if is_shared_memory_subclass(field_type):
            raise AttributeError(f"Cannot replace nested SharedMemoryObject '{name}'. Modify its attributes instead.")

        self._shared_dict[name] = value

    def to_dict(self):
        """Recursively converts the shared object to a regular dictionary."""
        plain_dict = {}
        for key in self.__class__.__annotations__:
            value = getattr(self, key)
            if isinstance(value, SharedMemoryObject):
                plain_dict[key] = value.to_dict()
            else:
                plain_dict[key] = value
        return plain_dict

    def __repr__(self):
        """Provides a nice, readable representation of the object's current state."""
        return f"{self.__class__.__name__}({self.to_dict()})"
