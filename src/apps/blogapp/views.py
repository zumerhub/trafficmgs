from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from django.views.generic import (
    ListView, 
    DetailView, 
    CreateView, 
    DeleteView, 
    UpdateView
)
 
from .models import Post
# from apps.blogapp.models import Post

# Create your views here.


def home(request):
    context = {
        'posts': Post.objects.all()
        }
    
    return render(request, 'blog/home.html', context)

# Setting a ListView to get Post-detail by post_id.
class PostListView(ListView):
    model = Post
    template_name = 'blog/home.html'
    context_object_name = 'posts'
    ordering = ['-date_posted']  # setting this to get a new list of posts at the top of the page
    paginate_by = 5

class UserPostListView(ListView):
    model = Post
    template_name = 'blog/user_posts.html'
    context_object_name = 'posts'
    paginate_by = 5
    
    def get_queryset(self):
        user = get_object_or_404(User, username=self.kwargs.get('username'))
        print(user)
        return Post.objects.filter(author=user).order_by('-date_posted')

class PostDetailView(DetailView):
    model = Post
    template_name = 'blog/post_detail.html'
    
    # setting the User (To Create a New post) with required login credentials using the (LoginRequiredMixin)
class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    fields = ['title', 'content']
    template_name = 'blog/post_form.html'
    
    # setting author as user to create the post
    def form_valid(self, form):
        form.instance.author = self.request.user        
        return super().form_valid(form)
    
    # setting the User to Login (To Update a post) using LoginRequiredMixin
class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    fields = ['title', 'content']
    template_name = 'blog/post_form.html'
    
    # setting author as user to create the post
    def form_valid(self, form):
        form.instance.author = self.request.user        
        return super().form_valid(form)
    
    # UserPassesTestMixin function  (To prevent anothers to redit-post ) using UserPassesTestMixin
    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False
    
    # setting the User to Login (To Delete a post) using LoginRequiredMixin
class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    template_name = 'blog/post_confirm_delete.html'
    success_url = '/'  # redirect to home page after deleting a post

    def test_func(self):
        post = self.get_object()
        return self.request.user == post.author
        
        # if self.request.user == post.author:
        #     return True
        # return False
    
    
    
def about(request):
    return render(request, 'blog/about.html', {"title": 'About'})