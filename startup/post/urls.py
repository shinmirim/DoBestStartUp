from django.urls import path
from . import views
urlpatterns = [
    path('product',views.product),
    path('image',views.image),
    path('review',views.review),
    path('selecttext', views.selecttext),
  
    
]
