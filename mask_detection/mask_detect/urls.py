from django.contrib import admin
from django.urls import path
from . import views
from .views import process_image
from .views import live_feed
from .views import live_construction

urlpatterns = [
    path('', views.index),
    path('upload/', process_image, name='upload-image'),
    path('live_feed/', live_feed, name='live_feed'),
    path('live_construction/', live_construction, name='live_construction'),
]