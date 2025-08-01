from django.contrib import admin
from django.contrib.auth.models import User
from django.utils.html import format_html
from .models import (
    QuizDomain, Quiz, Question,
    QuizSession, QuizAnswer,
    QADocument, RAGQuery
)

# Inline to show user's quiz sessions in User admin
class QuizSessionInline(admin.TabularInline):
    model = QuizSession
    extra = 0
    fields = ('quiz', 'percentage', 'status', 'get_grade', 'started_at', 'completed_at')
    readonly_fields = ('quiz', 'percentage', 'status', 'get_grade', 'started_at', 'completed_at')
    can_delete = False

    def get_grade(self, obj):
        return obj.get_grade()
    get_grade.short_description = 'Grade'

# Extend User admin to show email and quiz attempts
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'quiz_attempts_count')
    inlines = [QuizSessionInline]

    def quiz_attempts_count(self, obj):
        return obj.quiz_sessions.count()
    quiz_attempts_count.short_description = 'Quiz Attempts'

# Customize QuizSession admin
@admin.register(QuizSession)
class QuizSessionAdmin(admin.ModelAdmin):
    list_display = (
        'user_email', 'quiz', 'percentage', 'get_grade',
        'status', 'time_taken', 'started_at', 'completed_at'
    )
    list_filter = ('status', 'quiz__domain', 'started_at')
    search_fields = ('user__username', 'user__email', 'quiz__name')

    def user_email(self, obj):
        return obj.user.email
    user_email.short_description = 'User Email'

    def get_grade(self, obj):
        return obj.get_grade()
    get_grade.short_description = 'Grade'

# Register other models with basic admin
admin.site.register(QuizDomain)
admin.site.register(Quiz)
admin.site.register(Question)
admin.site.register(QuizAnswer)
admin.site.register(QADocument)
admin.site.register(RAGQuery)

# Unregister default User admin and register the customized one
from django.contrib.auth.admin import UserAdmin
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)
