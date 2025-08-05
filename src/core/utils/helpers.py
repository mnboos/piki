from multiprocessing import Condition, Event, Queue

class MultiprocessingDequeue:
    def __init__(self, queue: Queue) -> None:
        self.queue = queue
        self.condition = Condition()
        self.event = Event()

    def append(self, item):
        with self.condition:
            if self.queue.full():
                self.queue.get()
            self.queue.put(item)
            self.condition.notify_all()
        self.event.set()

    def wait_for_data(self):
        self.event.wait()
        self.event.clear()

    def popleft(self):
        with self.condition:
            if not self.queue.empty():
                return self.queue.get()
        return None