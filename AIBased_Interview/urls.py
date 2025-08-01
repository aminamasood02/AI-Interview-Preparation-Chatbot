from django.contrib import admin
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('test-csrf/', views.test_csrf, name='test_csrf'),
    path('homepage/', views.homepage, name='homepage'),
    path('login/', views.login, name='login'), 
    path('signup/', views.signup, name='signup'),
    path('prompt/', views.prompt, name='prompt'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('cv_based/', views.cv_based, name='cv_based'),
    path('open_ai_prompt/', views.open_ai_prompt, name='open_ai_prompt'),
    path('main_prompt/', views.main_prompt, name='main_prompt'), 
    path('generic_prompt/', views.generic_prompt, name='generic_prompt'),
    path('logout/', views.logout, name='logout'),
    path('confidence_checker/', views.confidence_checker, name='confidence_checker'),
    path('chat_support/', views.chat_support, name='chat_support'),
    
    path('password-reset/', auth_views.PasswordResetView.as_view(template_name='registration/password_reset.html'), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='registration/password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='registration/password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='registration/password_reset_complete.html'), name='password_reset_complete'),
    
    # Quiz System URLs
    path('quiz/', views.quiz_dashboard, name='quiz_dashboard'),
    path('quiz/domains/', views.quiz_domain_selection, name='quiz_domain_selection'),
    path('quiz/setup/<int:domain_id>/', views.quiz_setup, name='quiz_setup'),
    path('quiz/start/<int:quiz_id>/', views.quiz_start, name='quiz_start'),
    path('quiz/take/<int:session_id>/', views.quiz_take, name='quiz_take'),
    path('quiz/complete/<int:session_id>/', views.quiz_complete, name='quiz_complete'),
    path('quiz/result/<int:session_id>/', views.quiz_result, name='quiz_result'),
    path('quiz/history/', views.quiz_history, name='quiz_history'),
]