from django.contrib import admin
from .models import LearnerProfile, ChatSession, ChatMessage


@admin.register(LearnerProfile)
class LearnerProfileAdmin(admin.ModelAdmin):
    list_display  = ['learner_id', 'display_name', 'created_at']
    search_fields = ['learner_id', 'display_name']


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'learner', 'started_at', 'ended_at', 'total_turns']
    list_filter  = ['learner']


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['id', 'session', 'role', 'active_agent', 'timestamp', 'risk_score']
    list_filter  = ['role', 'active_agent']
