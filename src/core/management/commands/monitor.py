from django.core.management.base import BaseCommand
from ...utils.metrics import LiveMetricsDashboard, retrieve_queue

class Command(BaseCommand):
    help = 'Launches a real-time CLI dashboard to monitor worker processes.'

    def handle(self, *args, **options):
        queue = retrieve_queue()
        dashboard = LiveMetricsDashboard(queue)
        dashboard.run()