from django.urls import path
from . import views

urlpatterns = [
    # Pages
    path('',            views.onboarding,  name='onboarding'),
    path('chat/',       views.chat,        name='chat'),
    path('profile/',    views.profile,     name='profile'),
    path('dashboard/',  views.dashboard,   name='dashboard'),
    path('reading/',    views.reading,     name='reading'),

    # API – Chat / Session
    path('api/onboard/',                           views.api_onboard,        name='api_onboard'),
    path('api/chat/',                              views.api_chat,           name='api_chat'),
    path('api/profile/save/',                      views.api_profile_save,   name='api_profile_save'),
    path('api/session/<int:session_id>/history/',  views.api_session_history,     name='api_session_history'),
    path('api/session/end/',                       views.api_session_end,         name='api_session_end'),
    path('api/interventions/apply/',               views.api_interventions_apply, name='api_interventions_apply'),

    # API – IELTS Reading
    path('api/reading/upload/',       views.api_reading_upload,       name='api_reading_upload'),
    path('api/reading/next-section/', views.api_reading_next_section, name='api_reading_next_section'),
    path('api/reading/answer/',       views.api_reading_answer,       name='api_reading_answer'),
    path('api/reading/hint/',         views.api_reading_hint,         name='api_reading_hint'),
    path('api/reading/strategy/',     views.api_reading_strategy,     name='api_reading_strategy'),
    path('api/reading/assistant/',          views.api_reading_assistant,          name='api_reading_assistant'),
    path('api/reading/paragraph-strategy/', views.api_reading_paragraph_strategy, name='api_reading_paragraph_strategy'),
    path('api/reading/section-tips/', views.api_reading_section_tips, name='api_reading_section_tips'),

    # API - Passage.txt reading
    path('api/reading/paragraph/', views.api_reading_paragraph, name='api_reading_paragraph'),
]
