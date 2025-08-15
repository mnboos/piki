from django.core.management.base import BaseCommand
from multiprocessing import Event

import src.core.utils.shared
from ...utils.metrics import dashboard


class Command(BaseCommand):
    help = "Launches a real-time CLI dashboard to monitor worker processes."

    def handle(self, *args, **options):
        from ...utils import stream

        print("Streaming to: ", src.core.utils.common.output_buffer)
        dashboard.run()
