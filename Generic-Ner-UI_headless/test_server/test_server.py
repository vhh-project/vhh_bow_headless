import sys

from django.conf import settings
from django.core.management import execute_from_command_line
from django.http import JsonResponse
from django.urls import path

settings.configure(
    DEBUG=True,
    ROOT_URLCONF=sys.modules[__name__],
)


def bow_notification(request):
    print(request.POST)
    print(request.POST["task_id"])
    print(request.POST["status"])
    return JsonResponse({"received": "true"}, safe=False)


urlpatterns = [
    path("bow_notification", bow_notification, name="bow_notification"),
]

if __name__ == '__main__':
    execute_from_command_line(sys.argv)