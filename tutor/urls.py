from django.urls import path
from . import views

urlpatterns = [
    # Pages
    path('',            views.onboarding,  name='onboarding'),
    path('chat/',       views.chat,        name='chat'),
    path('profile/',    views.profile,     name='profile'),
    path('dashboard/',  views.dashboard,   name='dashboard'),
    path('reading/',    views.reading,     name='reading'),
    path('study-plan/', views.study_plan,  name='study_plan'),
    path('writing/',    views.writing,     name='writing'),
    path('speaking/',   views.speaking,    name='speaking'),
    path('listening/',  views.listening,   name='listening'),
    path('agent-workflow/', views.agent_workflow, name='agent_workflow'),

    # API – Chat / Session
    path('api/onboard/',                           views.api_onboard,        name='api_onboard'),
    path('api/chat/',                              views.api_chat,           name='api_chat'),
    path('api/chat/stream/',                       views.api_chat_stream,    name='api_chat_stream'),
    path('api/profile/save/',                      views.api_profile_save,   name='api_profile_save'),
    path('api/session/<int:session_id>/history/',  views.api_session_history,     name='api_session_history'),
    path('api/session/end/',                       views.api_session_end,         name='api_session_end'),
    path('api/interventions/apply/',               views.api_interventions_apply, name='api_interventions_apply'),
    path('api/study-plan/chat/',                   views.api_study_plan_chat,     name='api_study_plan_chat'),
    path('api/study-plan/generate/',               views.api_study_plan_generate, name='api_study_plan_generate'),
    path('api/dashboard/feedback/',                views.api_dashboard_feedback,  name='api_dashboard_feedback'),
    path('api/writing/generate/',                  views.api_writing_generate,    name='api_writing_generate'),
    path('api/writing/feedback/',                  views.api_writing_feedback,    name='api_writing_feedback'),
    path('api/writing/ielts-topic/',               views.api_writing_ielts_topic, name='api_writing_ielts_topic'),
    path('api/writing/stt/',                       views.api_writing_stt,         name='api_writing_stt'),
    path('api/speaking/generate/',                 views.api_speaking_generate,   name='api_speaking_generate'),
    path('api/speaking/chat/',                     views.api_speaking_chat,       name='api_speaking_chat'),
    path('api/listening/strategy/',                views.api_listening_strategy,  name='api_listening_strategy'),
    path('api/listening/logic-chain/',             views.api_listening_logic_chain, name='api_listening_logic_chain'),

    # API – IELTS Reading
    path('api/reading/upload/',       views.api_reading_upload,       name='api_reading_upload'),
    path('api/reading/next-section/', views.api_reading_next_section, name='api_reading_next_section'),
    path('api/reading/answer/',       views.api_reading_answer,       name='api_reading_answer'),
    path('api/reading/hint/',         views.api_reading_hint,         name='api_reading_hint'),
    path('api/reading/strategy/',     views.api_reading_strategy,     name='api_reading_strategy'),
    path('api/reading/assistant/',          views.api_reading_assistant,          name='api_reading_assistant'),
    path('api/reading/set-assistant-mode/', views.api_reading_set_assistant_mode, name='api_reading_set_assistant_mode'),
    path('api/reading/paragraph-strategy/', views.api_reading_paragraph_strategy, name='api_reading_paragraph_strategy'),
    path('api/reading/section-tips/', views.api_reading_section_tips, name='api_reading_section_tips'),

    # API - Passage.txt reading
    path('api/reading/paragraph/', views.api_reading_paragraph, name='api_reading_paragraph'),

    # Admin – Strategy Optimisation
    path('api/admin/strategy-performance/', views.api_strategy_performance, name='api_strategy_performance'),
]
