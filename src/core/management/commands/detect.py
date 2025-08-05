from django.core.management.base import BaseCommand
from multiprocessing import Event
from ...utils.metrics import dashboard


class Command(BaseCommand):
    help = 'Launches a real-time CLI dashboard to monitor worker processes.'

    def handle(self, *args, **options):
        from ...utils import stream
        print("Streaming to: ", stream.output_buffer)
        dashboard.run()
