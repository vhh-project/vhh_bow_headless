from django.urls import path, re_path
from . import views

"""Register urls here
"""

urlpatterns = [
    path("get_all_files", views.get_all_files, name="get_all_files"),
    re_path(r'^get_json/(?P<ids>[\w\-&]+)$', views.get_json, name="get_json"),
    path("delete_paths", views.delete_files, name="delete_files"),
    path("delete/<str:id>", views.delete, name="delete"),
    path("img/<str:id>/<int:page>.png", views.image, name="image"),
    path("ocr_data.zip", views.download_all, name="download_data"),
    path("<str:id>/result.pdf", views.download_pdf, name="download_pdf"),
    path("<str:id>/original", views.download_original, name="download_original"),
    path("upload", views.upload, name="upload"),
]
