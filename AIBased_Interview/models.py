# Enhanced models for comprehensive quiz system
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator

class QuizDomain(models.Model):
    """Available quiz domains/subjects"""
    DOMAIN_CHOICES = [
        ('python', 'Python Programming'),
        ('javascript', 'JavaScript'),
        ('java', 'Java Programming'),
        ('react', 'React.js'),
        ('django', 'Django Framework'),
        ('data_science', 'Data Science'),
        ('machine_learning', 'Machine Learning'),
        ('web_development', 'Web Development'),
        ('database', 'Database Management'),
        ('algorithms', 'Algorithms & Data Structures'),
        ('system_design', 'System Design'),
        ('networking', 'Computer Networks'),
        ('cybersecurity', 'Cybersecurity'),
        ('cloud_computing', 'Cloud Computing'),
        ('devops', 'DevOps'),
    ]
    
    name = models.CharField(max_length=50, choices=DOMAIN_CHOICES, unique=True)
    display_name = models.CharField(max_length=100)
    description = models.TextField()
    icon = models.CharField(max_length=50, default='fas fa-code')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.display_name
    
    class Meta:
        ordering = ['display_name']

class Quiz(models.Model):
    """Quiz sessions created by AI or manually"""
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ]
    
    name = models.CharField(max_length=255)
    description = models.TextField()
    domain = models.ForeignKey(QuizDomain, on_delete=models.CASCADE, related_name='quizzes')
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES, default='medium')
    total_questions = models.PositiveIntegerField()
    time_limit = models.PositiveIntegerField(help_text="Time limit in minutes", default=30)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    is_ai_generated = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.domain.display_name}"
    
    class Meta:
        ordering = ['-created_at']

class Question(models.Model):
    """Individual questions in a quiz"""
    quiz = models.ForeignKey(Quiz, related_name='questions', on_delete=models.CASCADE)
    question_text = models.TextField()
    option_a = models.CharField(max_length=500)
    option_b = models.CharField(max_length=500)
    option_c = models.CharField(max_length=500)
    option_d = models.CharField(max_length=500)
    correct_answer = models.CharField(max_length=1, choices=[('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D')])
    explanation = models.TextField(blank=True, help_text="Explanation for the correct answer")
    difficulty = models.CharField(max_length=10, choices=Quiz.DIFFICULTY_CHOICES, default='medium')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Q{self.id}: {self.question_text[:50]}..."
    
    class Meta:
        ordering = ['id']

class QuizSession(models.Model):
    """Individual quiz attempt by a user"""
    STATUS_CHOICES = [
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('abandoned', 'Abandoned'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='quiz_sessions')
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='sessions')
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='in_progress')
    score = models.PositiveIntegerField(default=0)
    total_questions = models.PositiveIntegerField()
    correct_answers = models.PositiveIntegerField(default=0)
    wrong_answers = models.PositiveIntegerField(default=0)
    percentage = models.FloatField(default=0.0, validators=[MinValueValidator(0), MaxValueValidator(100)])
    time_taken = models.PositiveIntegerField(default=0, help_text="Time taken in seconds")
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def save(self, *args, **kwargs):
        if self.total_questions > 0:
            self.percentage = (self.correct_answers / self.total_questions) * 100
        super().save(*args, **kwargs)
    
    def get_grade(self):
        if self.percentage >= 90:
            return 'A+'
        elif self.percentage >= 80:
            return 'A'
        elif self.percentage >= 70:
            return 'B'
        elif self.percentage >= 60:
            return 'C'
        elif self.percentage >= 50:
            return 'D'
        else:
            return 'F'
    
    def __str__(self):
        return f"{self.user.username} - {self.quiz.name} ({self.status})"
    
    class Meta:
        ordering = ['-started_at']

class QuizAnswer(models.Model):
    """User's answers to quiz questions"""
    session = models.ForeignKey(QuizSession, on_delete=models.CASCADE, related_name='answers')
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    selected_answer = models.CharField(max_length=1, choices=[('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D')])
    is_correct = models.BooleanField(default=False)
    time_taken = models.PositiveIntegerField(default=0, help_text="Time taken for this question in seconds")
    answered_at = models.DateTimeField(auto_now_add=True)
    
    def save(self, *args, **kwargs):
        self.is_correct = (self.selected_answer == self.question.correct_answer)
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.session.user.username} - Q{self.question.id} - {self.selected_answer}"
    
    class Meta:
        unique_together = ['session', 'question']
        ordering = ['question__id']

class QADocument(models.Model):
    """Model to store Q&A documents for RAG functionality"""
    question = models.TextField()
    answer = models.TextField()
    combined_text = models.TextField()  # For embedding generation
    embedding_index = models.IntegerField(null=True, blank=True)  # Index in FAISS
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Q: {self.question[:50]}..."
    
    class Meta:
        ordering = ['-created_at']

class RAGQuery(models.Model):
    """Model to store user queries and responses for RAG"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='rag_queries')
    query = models.TextField()
    response = models.TextField()
    relevant_documents = models.JSONField(default=list)  # Store retrieved document IDs
    similarity_scores = models.JSONField(default=list)  # Store similarity scores
    processing_time = models.FloatField(default=0.0)  # Time taken to process query
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}: {self.query[:50]}..."
    
    class Meta:
        ordering = ['-created_at'] 