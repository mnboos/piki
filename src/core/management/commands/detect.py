from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Launches a real-time CLI dashboard to monitor worker processes."

    def handle(self, *args, **options):
        from ...utils.stream import dashboard

        dashboard.run()
