 # Profile URLs
from django.urls import path
from django.contrib.auth.views import (
    PasswordResetView, PasswordResetDoneView, PasswordResetCompleteView, PasswordResetConfirmView
) 
from . import views

urlpatterns = [
    path('profile/', views.profile, name='profile'),  # Profile page
    path('logout/', views.logout_view, name='logout'),  # Logout page
    path('login/', views.login_view, name='login'),  # Login page
    path('register/', views.register, name='register'),  # Registration page
   
   
   # Password reset flow
    path('password-reset/', PasswordResetView.as_view(), name='password_reset'),  # Password reset page
    path('password-reset/done/', PasswordResetDoneView.as_view(), name='password_reset_done'),  # Password reset done page
    path('password-reset-confirm/<uidb64>/<token>/', PasswordResetConfirmView.as_view(), name='password_reset_confirm'),  # Password reset confirm page
    path('password-reset-complete/', PasswordResetCompleteView.as_view(), name='password_reset_complete'),  # Password reset complete page
    
   ]