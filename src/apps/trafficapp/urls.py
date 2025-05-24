from django.urls import path
from . import views
app_name = 'traffic'

urlpatterns = [
    path('', views.index, name='index'),
    path('traffic/', views.DashboardView.as_view(), name='dashboard'),

    # API endpoints 
    path('api/traffic-data/', views.traffic_data_api, name='traffic_data_api'),
    
    # Administrative actions
    path('api/reset-counters/', views.reset_counters, name='reset_counters'),

]
