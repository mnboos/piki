import errno

from rich.console import group
from rich.panel import Panel

from .shared import MultiprocessingDequeue
from rich.live import Live
from rich.table import Table
import atexit
import time

from multiprocessing import Queue, Manager
from multiprocessing.managers import BaseManager


class QueueManager(BaseManager):
    pass


_manager_getter_name = "get_dashboard_queue"
QUEUE_KEY = b"abcd-foobar!!"

# m = QueueManager()
queue = Queue(10)


QueueManager.register(_manager_getter_name, callable=lambda: queue)
queue_manager = QueueManager(address=("127.0.0.1", 50000), authkey=QUEUE_KEY)


def cleanup():
    print("[Metrics] Shutting down...")
    queue_manager.shutdown()
    print("[Metrics] Shut down")


# @atexit.register
# def cleanup():
#     if hasattr(queue_manager, "shutdown"):
#         queue_manager.shutdown()


def retrieve_queue(max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            # m = QueueManager(address=("127.0.0.1", 50000), authkey=QUEUE_KEY)
            queue_manager.connect()

            return getattr(queue_manager, _manager_getter_name)()
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    "Address is already in use, starting queue manager is not possible."
                ) from e
        except (EOFError, ConnectionRefusedError) as e:
            retries += 1
            if retries > max_retries:
                raise RuntimeError(
                    "Connection to the queue-manager could not be established. Is the application running?"
                ) from e

            backoff = min(30, (2**retries) * 0.1)
            print(
                f"Queue manager not yet started, trying again in {str(backoff).rjust(5)}s (retry {retries})"
            )

            time.sleep(backoff)
            continue
    return None


class LiveMetricsDashboard:
    def __init__(self) -> None:
        queue = retrieve_queue(max_retries=100)

        self.queue = MultiprocessingDequeue(queue)
        self.state: dict[int, dict] = {}

    def update(self, *, worker_id: int, inference_time: float) -> None:
        self.queue.append((worker_id, inference_time))

    @group()
    def get_panels(self):
        yield Panel("Hello", style="on blue")
        yield Panel("World", style="on red")

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
        # m.start()
        with Live(self._generate_table(), screen=False) as live:
            # with Live(Panel(self.get_panels()), screen=False) as live:
            while True:
                try:
                    worker_id, inference_time = self.queue.queue.get()
                except EOFError:
                    self.queue.queue = retrieve_queue()
                # print("got data", d)
                # worker_id, inference_time = self.queue.popleft()

                self.state[worker_id] = {
                    "inference_time": str(inference_time),
                    "items": 1,
                }

                live.update(self._generate_table())
