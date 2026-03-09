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


# ── IELTS Reading ─────────────────────────────────────────────────

class IELTSPassage(models.Model):
    learner      = models.ForeignKey(LearnerProfile, on_delete=models.CASCADE, related_name='passages')
    title        = models.CharField(max_length=300, blank=True, default='')
    raw_text     = models.TextField()
    created_at   = models.DateTimeField(auto_now_add=True)
    is_active    = models.BooleanField(default=True)

    def __str__(self):
        return f"Passage {self.id}: {self.title or '(untitled)'}"


class IELTSSection(models.Model):
    passage      = models.ForeignKey(IELTSPassage, on_delete=models.CASCADE, related_name='sections')
    order        = models.PositiveSmallIntegerField()
    heading      = models.CharField(max_length=300, blank=True, default='')
    body         = models.TextField()
    image_path   = models.CharField(max_length=500, blank=True, default='')

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"Section {self.order} of Passage {self.passage_id}"


class IELTSQuestion(models.Model):
    passage      = models.ForeignKey(IELTSPassage, on_delete=models.CASCADE, related_name='questions')
    section      = models.ForeignKey(IELTSSection, on_delete=models.SET_NULL, null=True, blank=True, related_name='questions')
    order        = models.PositiveSmallIntegerField()
    text         = models.TextField()

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"Q{self.order}: {self.text[:60]}"


class ReadingAttempt(models.Model):
    learner       = models.ForeignKey(LearnerProfile, on_delete=models.CASCADE, related_name='reading_attempts')
    passage       = models.ForeignKey(IELTSPassage, on_delete=models.CASCADE, related_name='attempts')
    current_section_order = models.PositiveSmallIntegerField(default=1)
    completed     = models.BooleanField(default=False)
    score         = models.FloatField(null=True, blank=True)
    answers       = models.JSONField(default=dict)
    hints_used    = models.IntegerField(default=0)
    strategy_log  = models.JSONField(default=list)
    started_at    = models.DateTimeField(auto_now_add=True)
    updated_at    = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Attempt by {self.learner.display_name} on Passage {self.passage_id}"
