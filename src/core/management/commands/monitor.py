from django.core.management.base import BaseCommand
from ...utils.metrics import LiveMetricsDashboard


class Command(BaseCommand):
    help = "Launches a real-time CLI dashboard to monitor worker processes."

    def handle(self, *args, **options):
        dashboard = LiveMetricsDashboard()
        dashboard.run()
