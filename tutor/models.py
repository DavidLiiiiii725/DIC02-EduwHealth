from django.db import models


class LearnerProfile(models.Model):
    learner_id         = models.CharField(max_length=100, unique=True)
    display_name       = models.CharField(max_length=100, default='Learner')
    ld_confirmed       = models.JSONField(default=list)
    ld_suspected       = models.JSONField(default=list)
    ld_severity        = models.JSONField(default=dict)
    wm_span            = models.FloatField(default=5.0)
    attention_min      = models.FloatField(default=15.0)
    initiation_latency = models.FloatField(default=20.0)
    frustration_thresh = models.FloatField(default=0.55)
    created_at         = models.DateTimeField(auto_now_add=True)
    updated_at         = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.display_name} ({self.learner_id})"


class ChatSession(models.Model):
    learner     = models.ForeignKey(LearnerProfile, on_delete=models.CASCADE, related_name='sessions')
    started_at  = models.DateTimeField(auto_now_add=True)
    ended_at    = models.DateTimeField(null=True, blank=True)
    total_turns = models.IntegerField(default=0)

    def __str__(self):
        return f"Session {self.id} — {self.learner.display_name}"


class ChatMessage(models.Model):
    ROLE_CHOICES = [('user', 'User'), ('assistant', 'Assistant')]
    session      = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role         = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content      = models.TextField()
    active_agent = models.CharField(max_length=50, blank=True)
    timestamp    = models.DateTimeField(auto_now_add=True)
    wm_load      = models.FloatField(null=True)
    motivation   = models.FloatField(null=True)
    affect       = models.FloatField(null=True)
    fatigue      = models.FloatField(null=True)
    risk_score   = models.FloatField(null=True)
    risk_level   = models.CharField(max_length=20, blank=True)

    class Meta:
        ordering = ['timestamp']
