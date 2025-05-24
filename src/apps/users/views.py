from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm
from django.contrib.auth.views import PasswordResetView, PasswordResetConfirmView

# Create your views here.

import logging
logger = logging.getLogger(__name__)
def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            messages.success(request, f'Your account has been successfully created!. You are now logged in as {username}!')
            form.save()
            return redirect('login')
    else:
        form = UserRegisterForm() 
    return render(request, 'users/register.html', {'form': form})

@login_required
def profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, f'Your account has been updated successfully!!!')
            return redirect('profile')

            
    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)
    
    context = {
        'u_form': u_form,
        'p_form': p_form
    }
    return render(request, 'users/profile.html', context)

def login_view(request):
    form = AuthenticationForm(data=request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f'Welcome, {user.username}!')
            return redirect('profile')
        else:
            messages.error(request, 'Invalid username or password.')

    context = {
        'form': form,
        'title': 'Login',
    }
    return render(request, 'users/login.html', context)

def logout_view(request):
    logout(request)
    return render(request, 'users/logout.html', {'title': 'logout'})



class CustomPasswordResetView(PasswordResetView):
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['domain'] = '127.0.0.1:8000'
        context['protocol'] = 'https'  # or 'http' for development
        return context
    
    def form_valid(self, form):
        # Check the uidb64 and token before sending the email
        user = form.get_users(form.cleaned_data["email"]).first()
        if user:
            # You can print these to the console for debugging
            print(f"UIDB64: {user.pk}, Token: {user.get_reset_token()}")  # or however you handle token generation
        return super().form_valid(form)
    
class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    def form_invalid(self, form):
        logger.error("Form errors: %s", form.errors)
        return super().form_invalid(form)