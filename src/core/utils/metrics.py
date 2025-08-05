import multiprocessing.managers
from typing import Callable

from .helpers import MultiprocessingDequeue
from ctypes import c_float
from rich.live import Live
from rich.table import Table
import atexit

from multiprocessing import Queue
from multiprocessing.managers import BaseManager

class QueueManager(BaseManager):
    pass

QUEUE_KEY = b"abcd-foobar!!"

_queue = Queue(20)
QueueManager.register("get_dashboard_queue", callable=lambda: _queue)


def create_queue_manager():
    m = QueueManager(address=('', 50000), authkey=QUEUE_KEY)
    # s = m.get_server()
    # s.serve_forever()
    m.start()

    def cleanup():
        print("Cleaning up...")
        m.shutdown()

    atexit.register(cleanup)



def retrieve_queue():
    m = QueueManager(address=("127.0.0.1", 50000), authkey=QUEUE_KEY)
    m.connect()
    return m.get_dashboard_queue()

class LiveMetricsDashboard:
    def __init__(self, queue: Queue) -> None:

        self.queue = MultiprocessingDequeue(queue)
        self.state: dict[int, dict] = {}

    def update(self, *, worker_id: int, inference_time: float) -> None:
        self.queue.append((worker_id, inference_time))

    def _generate_table(self) -> Table:
        """Draws a Rich Table from the current dashboard_data."""
        table = Table(title="Multi-Process Inference Monitor")
        table.add_column("Worker ID", style="cyan")
        table.add_column("Inference Time (ms)", justify="right", style="magenta")
        table.add_column("Items Processed", justify="right", style="green")

        for worker_id, data in self.state.items():
            table.add_row(
                f"Worker {worker_id}",
                str(data["inference_time"]),
            )
        return table

    def run(self):
        with Live(self._generate_table(), screen=False) as live:
            while True:
                worker_id, inference_time = self.queue.queue.get()
                # print("got data", d)
                # worker_id, inference_time = self.queue.popleft()

                self.state[worker_id] = {
                        "inference_time": str(inference_time),
                        "items": 1,
                    }

                live.update(self._generate_table())