from django.contrib import admin
from django.urls import path
from .views import main, answer

urlpatterns = [
   path("", main),
   path("getAnswer", answer)
]
