from multiprocessing import Condition, Event, Queue
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
