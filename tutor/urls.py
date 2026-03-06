from django.urls import path
from . import views

urlpatterns = [
    # Pages
    path('',            views.onboarding,  name='onboarding'),
    path('chat/',       views.chat,        name='chat'),
    path('profile/',    views.profile,     name='profile'),
    path('dashboard/',  views.dashboard,   name='dashboard'),

    # API
    path('api/onboard/',                           views.api_onboard,        name='api_onboard'),
    path('api/chat/',                              views.api_chat,           name='api_chat'),
    path('api/profile/save/',                      views.api_profile_save,   name='api_profile_save'),
    path('api/session/<int:session_id>/history/',  views.api_session_history,name='api_session_history'),
    path('api/session/end/',                       views.api_session_end,    name='api_session_end'),
]
