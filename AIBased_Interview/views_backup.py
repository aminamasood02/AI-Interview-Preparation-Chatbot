from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect  
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.db.models import Q, Count, Avg, Sum
from django.core.paginator import Paginator
import google.generativeai as genai
import json
import docx
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Import the new models
# from .models import (
#     QuizCategory, AIQuiz, AIQuestion, QuizAttempt, 
#     QuizAnswer, UserProgress, QuizFeedback
# )

def index(request):
    return render(request, 'index.html')

@login_required(login_url='/') 
def homepage(request):
    return render(request, 'homepage.html')

@login_required(login_url='/') 
def prompt(request):
    return render(request, 'prompt.html')

@login_required(login_url='/') 
def about(request):
    return render(request, 'About Us.html')

@login_required(login_url='/') 
def contact(request):
    return render(request, 'contact.html')

@login_required(login_url='/') 
def cv_based(request):
    if request.method == 'POST':
        try:
            if 'cv_file' not in request.FILES:
                return JsonResponse({'error': 'No file uploaded'}, status=400)
            
            cv_file = request.FILES['cv_file']
            
            # Check if file is .docx
            if not cv_file.name.lower().endswith('.docx'):
                return JsonResponse({'error': 'Only .docx files are allowed'}, status=400)
            
            # Save file temporarily
            file_name = default_storage.save(f'temp_cv_{request.user.id}.docx', ContentFile(cv_file.read()))
            file_path = default_storage.path(file_name)
            
            try:
                # Extract text from CV
                cv_text = extract_text_from_docx(file_path)
                
                if not cv_text.strip():
                    return JsonResponse({'error': 'Could not extract text from CV. Please ensure the file contains text.'}, status=400)
                
                # Generate interview questions
                questions = generate_questions_with_gemini(cv_text)
                
                # Clean up temporary file
                default_storage.delete(file_name)
                
                return JsonResponse({'questions': questions})
                
            except Exception as e:
                # Clean up temporary file in case of error
                if default_storage.exists(file_name):
                    default_storage.delete(file_name)
                raise e
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return render(request, 'cv_based.html')

@login_required(login_url='/') 
def open_ai_prompt(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            prompt = data.get('prompt', '').strip()
            language = data.get('language', 'python')
            
            if not prompt:
                return JsonResponse({'error': 'Prompt is required'}, status=400)
            
            # Generate code using Gemini API
            code = generate_code(prompt, language)
            return JsonResponse({'code': code})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return render(request, 'open_ai_prompt.html')

@login_required(login_url='/') 
def main_prompt(request): 
    return render(request, 'main_prompt.html')

@login_required(login_url='/') 
def generic_prompt(request):
    return HttpResponse("This is the generic prompt page.")

@csrf_protect 
def login(request):
    print(f"Login view accessed: Method={request.method}, GET params={request.GET}, POST params={request.POST}")
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            users = User.objects.filter(email=email)  # Use filter instead of get()
            if users.count() == 1:  # Ensure there's exactly one user
                user = users.first()
                user = authenticate(request, username=user.username, password=password)

                if user is not None:
                    auth_login(request, user)
                    # Check if there's a 'next' parameter and redirect accordingly
                    next_url = request.GET.get('next') or request.POST.get('next')
                    if next_url:
                        return redirect(next_url)
                    return redirect('homepage')  
                else:
                    messages.error(request, '❌ Invalid email or password. Please try again.')
            else:
                messages.error(request, '❌ Multiple accounts found with this email. Please contact support.')
        except User.DoesNotExist:
            messages.error(request, '❌ No account found with this email.')
        
        return redirect('index') 

    return render(request, 'index.html')

# @csrf_protect  
def signup(request):
    if request.method == 'POST':
        email = request.POST.get('email').strip()
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        if not email or not password or not confirm_password:
            messages.error(request, '❌ All fields are required.')
            return redirect('index')

        if password != confirm_password:
            messages.error(request, '❌ Passwords do not match. Please try again.')
            return redirect('index')

        if User.objects.filter(email=email).exists():
            messages.error(request, '❌ An account with this email already exists.')
            return redirect('index')


        user = User.objects.create_user(username=email, email=email, password=password)
        user = authenticate(request, username=email, password=password)
        if user:
            auth_login(request, user)
            messages.success(request, '✅ Account created successfully! You are now logged in.')
            return redirect('index')

    return render(request, 'index.html')

@login_required(login_url='/')
def logout(request):
    auth_logout(request)
    messages.success(request, '✅ Logged out successfully!')
    return redirect('index')

def generate_code(prompt: str, language: str) -> str:
    """Generate code using Gemini API"""
    api_key = 'AIzaSyDI8eZiJhgKlPfoxhSI_88u-6kEnrgOsyg'
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    full_prompt = f"Write {language} code to {prompt}. Return only the complete, functional code with no explanations, comments, or markdown formatting."
    
    response = model.generate_content(full_prompt)
    code_text = (
        response.text
        .replace("```", "")
        .replace("'''", "")
        .replace(f"```{language}", "")
        .strip()
    )
    return code_text

def extract_text_from_docx(file_path):
    """Extract text from .docx file"""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def generate_questions_with_gemini(cv_text, num_questions=20):
    """Generate interview questions using Gemini API"""
    api_key = 'AIzaSyDI8eZiJhgKlPfoxhSI_88u-6kEnrgOsyg'
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    prompt = f"""
    Based on the following CV, generate {num_questions} interview-style questions
    that could be asked about the candidate, focusing on their skills, experience, and background.

    Return a JSON-style dictionary where each key is a clearly phrased question
    and each corresponding value is the best possible answer based on the CV.
    Only return the dictionary, no explanation, no markdown.

    CV:
    {cv_text}
    """
    
    response = model.generate_content(prompt)
    
    # Clean the output
    cleaned_questions = (
        response.text
        .replace("```", "")
        .replace("json", "")
        .replace("'''", "")
        .strip()
    )
    
    try:
        # Try to parse as JSON to validate format
        questions_dict = json.loads(cleaned_questions)
        return questions_dict
    except json.JSONDecodeError:
        # If JSON parsing fails, return the raw text
        return {"response": cleaned_questions}

@login_required(login_url='/') 
def confidence_checker(request):
    if request.method == 'POST':
        try:
            print(f"Confidence checker called by user: {request.user.username}")
            
            if 'confidence_image' not in request.FILES:
                return JsonResponse({'error': 'No image uploaded'}, status=400)
            
            image_file = request.FILES['confidence_image']
            print(f"Image file received: {image_file.name}, size: {image_file.size}")
            
            # Check if file is an image
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            if not any(image_file.name.lower().endswith(ext) for ext in allowed_extensions):
                return JsonResponse({'error': 'Only image files are allowed (.jpg, .jpeg, .png, .bmp, .gif)'}, status=400)
            
            # Save file temporarily
            file_name = default_storage.save(f'temp_confidence_{request.user.id}.jpg', ContentFile(image_file.read()))
            file_path = default_storage.path(file_name)
            print(f"Image saved temporarily at: {file_path}")
            
            try:
                # Load model and predict
                print("Loading confidence model...")
                model = initialize_confidence_model()
                print("Model loaded, making prediction...")
                confidence_result = predict_confidence(file_path, model)
                print(f"Prediction result: {confidence_result}")
                
                # Clean up temporary file
                default_storage.delete(file_name)
                print("Temporary file cleaned up")
                
                return JsonResponse({
                    'result': confidence_result,
                    'message': f'Analysis complete! You appear to be {confidence_result}.',
                    'advice': get_confidence_advice(confidence_result)
                })
                
            except Exception as e:
                # Clean up temporary file in case of error
                if default_storage.exists(file_name):
                    default_storage.delete(file_name)
                print(f"Error during model processing: {str(e)}")
                raise e
                
        except Exception as e:
            print(f"Error in confidence_checker: {str(e)}")
            return JsonResponse({'error': f'Processing error: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def initialize_confidence_model():
    """Initialize and load the confidence detection model"""
    try:
        device = torch.device("cpu")  # Use CPU for compatibility
        
        # Initialize the same model architecture as in training
        # Using weights=None instead of pretrained=False (deprecated)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        # Load the trained weights
        model_path = os.path.join(os.path.dirname(__file__), '..', 'confidence_model.pth')
        if not os.path.exists(model_path):
            # Try alternative path
            model_path = 'confidence_model.pth'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

def predict_confidence(image_path, model):
    """Predict confidence from image using the trained model"""
    try:
        print(f"Making prediction for image: {image_path}")
        
        # Preprocess the image
        image_tensor = preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence_score = probabilities[predicted_class].item()
            
            print(f"Prediction probabilities: {probabilities}")
            print(f"Predicted class: {predicted_class}, confidence: {confidence_score:.4f}")
            
            # Map class to label (0: confident, 1: unconfident based on training)
            result = "confident" if predicted_class == 0 else "unconfident"
            print(f"Final result: {result}")
            
            return result
            
    except Exception as e:
        print(f"Error in predict_confidence: {str(e)}")
        raise e

def get_confidence_advice(confidence_result):
    """Get advice based on confidence result"""
    if confidence_result == "confident":
        return [
            "Great! You're showing positive confident body language.",
            "Keep maintaining good eye contact and posture.",
            "Practice speaking clearly and at a moderate pace.",
            "Remember to smile naturally during interviews."
        ]
    else:
        return [
            "Work on maintaining better posture - sit/stand straight.",
            "Practice making eye contact - look directly at the camera.",
            "Try to relax your facial muscles and smile naturally.",
            "Practice deep breathing to reduce anxiety before interviews.",
            "Record yourself practicing interview questions to improve."
        ]

def preprocess_image(image_path):
    """Preprocess image for the confidence model"""
    try:
        print(f"Preprocessing image: {image_path}")
        
        # Load and preprocess the image
        image = Image.open(image_path)
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"Converted image from {image.mode} to RGB")
        
        # Define the same transforms as in training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        print(f"Image preprocessed successfully, tensor shape: {image_tensor.shape}")
        return image_tensor
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise e

# ============== PROGRESS TRACKING & QUIZ SYSTEM VIEWS ==============

@login_required(login_url='/')
def progress_tracking_dashboard(request):
    """Main progress tracking dashboard"""
    try:
        # Get or create progress records for all categories
        categories = QuizCategory.objects.filter(is_active=True)
        progress_data = []
        
        for category in categories:
            progress, created = UserProgress.objects.get_or_create(
                user=request.user,
                category=category
            )
            if not created:
                progress.update_progress()  # Update with latest data
            progress_data.append(progress)
        
        # Get overall statistics
        total_quizzes_taken = QuizAttempt.objects.filter(user=request.user, completed=True).count()
        total_xp = sum(p.xp_points for p in progress_data)
        
        # Get recent quiz attempts
        recent_attempts = QuizAttempt.objects.filter(
            user=request.user, 
            completed=True
        ).order_by('-completed_at')[:5]
        
        # Get recommended quizzes (based on user's weakest categories)
        weak_categories = sorted(progress_data, key=lambda x: x.average_score)[:3]
        recommended_quizzes = []
        for progress in weak_categories:
            quizzes = AIQuiz.objects.filter(
                category=progress.category,
                is_active=True,
                difficulty=progress.level
            ).exclude(
                quiz_attempts__user=request.user,
                quiz_attempts__completed=True
            )[:2]
            recommended_quizzes.extend(quizzes)
        
        context = {
            'progress_data': progress_data,
            'total_quizzes_taken': total_quizzes_taken,
            'total_xp': total_xp,
            'recent_attempts': recent_attempts,
            'recommended_quizzes': recommended_quizzes,
        }
        
        return render(request, 'progress_tracking/dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading dashboard: {str(e)}')
        return redirect('homepage')

@login_required(login_url='/')
def quiz_categories(request):
    """Display all quiz categories"""
    categories = QuizCategory.objects.filter(is_active=True)
    
    # Add progress info to each category
    categories_with_progress = []
    for category in categories:
        progress, _ = UserProgress.objects.get_or_create(
            user=request.user,
            category=category
        )
        category.user_progress = progress
        categories_with_progress.append(category)
    
    context = {
        'categories': categories_with_progress,
    }
    
    return render(request, 'progress_tracking/categories.html', context)

@login_required(login_url='/')
def category_quizzes(request, category_name):
    """Display quizzes for a specific category"""
    try:
        category = get_object_or_404(QuizCategory, name=category_name, is_active=True)
        
        # Get user's progress for this category
        progress, _ = UserProgress.objects.get_or_create(
            user=request.user,
            category=category
        )
        
        # Get all quizzes for this category
        quizzes = AIQuiz.objects.filter(category=category, is_active=True)
        
        # Add attempt info to each quiz
        quizzes_with_attempts = []
        for quiz in quizzes:
            user_attempts = QuizAttempt.objects.filter(
                user=request.user,
                quiz=quiz,
                completed=True
            ).order_by('-completed_at')
            
            quiz.user_attempts = user_attempts
            quiz.best_score = user_attempts.aggregate(
                best=models.Max('score')
            )['best'] or 0
            quiz.attempt_count = user_attempts.count()
            quiz.last_attempt = user_attempts.first()
            
            quizzes_with_attempts.append(quiz)
        
        context = {
            'category': category,
            'progress': progress,
            'quizzes': quizzes_with_attempts,
        }
        
        return render(request, 'progress_tracking/category_quizzes.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading category: {str(e)}')
        return redirect('quiz_categories')

@login_required(login_url='/')
def generate_ai_quiz(request, category_name):
    """Generate a new AI quiz for a category"""
    if request.method == 'POST':
        try:
            category = get_object_or_404(QuizCategory, name=category_name, is_active=True)
            
            # Get form data
            difficulty = request.POST.get('difficulty', 'beginner')
            num_questions = int(request.POST.get('num_questions', 10))
            time_limit = int(request.POST.get('time_limit', 30))
            
            # Generate quiz using AI
            quiz_data = generate_quiz_with_ai(category, difficulty, num_questions)
            
            if not quiz_data:
                messages.error(request, 'Failed to generate quiz. Please try again.')
                return redirect('category_quizzes', category_name=category_name)
            
            # Create quiz
            quiz = AIQuiz.objects.create(
                title=quiz_data['title'],
                category=category,
                difficulty=difficulty,
                description=quiz_data['description'],
                total_questions=num_questions,
                time_limit=time_limit,
                created_by=request.user,
                generation_prompt=quiz_data.get('prompt', '')
            )
            
            # Create questions
            for i, question_data in enumerate(quiz_data['questions'], 1):
                AIQuestion.objects.create(
                    quiz=quiz,
                    question_text=question_data['question'],
                    option_a=question_data['options']['A'],
                    option_b=question_data['options']['B'],
                    option_c=question_data['options']['C'],
                    option_d=question_data['options']['D'],
                    correct_answer=question_data['correct_answer'],
                    explanation=question_data.get('explanation', ''),
                    difficulty=difficulty,
                    question_order=i
                )
            
            messages.success(request, f'Quiz "{quiz.title}" generated successfully!')
            return redirect('take_quiz', quiz_id=quiz.id)
            
        except Exception as e:
            messages.error(request, f'Error generating quiz: {str(e)}')
            return redirect('category_quizzes', category_name=category_name)
    
    # GET request - show generation form
    category = get_object_or_404(QuizCategory, name=category_name, is_active=True)
    context = {
        'category': category,
        'difficulty_choices': [('beginner', 'Beginner'), ('intermediate', 'Intermediate'), 
                             ('advanced', 'Advanced'), ('expert', 'Expert')],
    }
    
    return render(request, 'progress_tracking/generate_quiz.html', context)

@login_required(login_url='/')
def take_quiz(request, quiz_id):
    """Take a quiz"""
    try:
        quiz = get_object_or_404(AIQuiz, id=quiz_id, is_active=True)
        
        # Check if user has an ongoing attempt
        ongoing_attempt = QuizAttempt.objects.filter(
            user=request.user,
            quiz=quiz,
            completed=False
        ).first()
        
        if request.method == 'POST':
            if request.POST.get('action') == 'start':
                # Start new quiz attempt
                if ongoing_attempt:
                    ongoing_attempt.delete()  # Clear any ongoing attempt
                
                attempt = QuizAttempt.objects.create(
                    user=request.user,
                    quiz=quiz,
                    total_questions=quiz.questions.count()
                )
                
                return redirect('quiz_question', attempt_id=attempt.id, question_num=1)
        
        # GET request - show quiz info
        user_attempts = QuizAttempt.objects.filter(
            user=request.user,
            quiz=quiz,
            completed=True
        ).order_by('-completed_at')
        
        context = {
            'quiz': quiz,
            'user_attempts': user_attempts,
            'ongoing_attempt': ongoing_attempt,
            'best_score': user_attempts.aggregate(best=models.Max('score'))['best'] or 0,
        }
        
        return render(request, 'progress_tracking/take_quiz.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading quiz: {str(e)}')
        return redirect('quiz_categories')

@login_required(login_url='/')
def quiz_question(request, attempt_id, question_num):
    """Display a specific question during quiz"""
    try:
        attempt = get_object_or_404(
            QuizAttempt, 
            id=attempt_id, 
            user=request.user, 
            completed=False
        )
        
        questions = attempt.quiz.questions.all()
        
        if question_num > questions.count():
            # Quiz completed
            return redirect('complete_quiz', attempt_id=attempt_id)
        
        question = questions[question_num - 1]
        
        if request.method == 'POST':
            selected_answer = request.POST.get('answer')
            
            if selected_answer:
                # Save answer
                quiz_answer, created = QuizAnswer.objects.get_or_create(
                    attempt=attempt,
                    question=question,
                    defaults={'selected_answer': selected_answer}
                )
                
                if not created:
                    quiz_answer.selected_answer = selected_answer
                    quiz_answer.save()
                
                # Move to next question
                next_question = question_num + 1
                if next_question <= questions.count():
                    return redirect('quiz_question', attempt_id=attempt_id, question_num=next_question)
                else:
                    return redirect('complete_quiz', attempt_id=attempt_id)
        
        # Get user's previous answer for this question
        previous_answer = QuizAnswer.objects.filter(
            attempt=attempt,
            question=question
        ).first()
        
        # Calculate progress
        answered_count = QuizAnswer.objects.filter(attempt=attempt).count()
        progress_percentage = (answered_count / questions.count()) * 100
        
        context = {
            'attempt': attempt,
            'question': question,
            'question_num': question_num,
            'total_questions': questions.count(),
            'previous_answer': previous_answer,
            'progress_percentage': progress_percentage,
            'time_limit_minutes': attempt.quiz.time_limit,
        }
        
        return render(request, 'progress_tracking/quiz_question.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading question: {str(e)}')
        return redirect('progress_tracking_dashboard')

@login_required(login_url='/')
def complete_quiz(request, attempt_id):
    """Complete quiz and show results"""
    try:
        attempt = get_object_or_404(
            QuizAttempt, 
            id=attempt_id, 
            user=request.user, 
            completed=False
        )
        
        # Calculate results
        answers = QuizAnswer.objects.filter(attempt=attempt)
        attempt.correct_answers = answers.filter(is_correct=True).count()
        attempt.complete_quiz()
        
        # Update user progress
        progress, _ = UserProgress.objects.get_or_create(
            user=request.user,
            category=attempt.quiz.category
        )
        progress.update_progress()
        
        # Generate AI feedback
        feedback = generate_ai_feedback(attempt)
        if feedback:
            QuizFeedback.objects.create(
                attempt=attempt,
                strengths=feedback['strengths'],
                weaknesses=feedback['weaknesses'],
                recommendations=feedback['recommendations'],
                next_steps=feedback['next_steps']
            )
        
        return redirect('quiz_results', attempt_id=attempt_id)
        
    except Exception as e:
        messages.error(request, f'Error completing quiz: {str(e)}')
        return redirect('progress_tracking_dashboard')

@login_required(login_url='/')
def quiz_results(request, attempt_id):
    """Show detailed quiz results"""
    try:
        attempt = get_object_or_404(
            QuizAttempt, 
            id=attempt_id, 
            user=request.user, 
            completed=True
        )
        
        # Get all answers with question details
        answers = QuizAnswer.objects.filter(attempt=attempt).select_related('question')
        
        # Get feedback if available
        feedback = getattr(attempt, 'feedback', None)
        
        # Get user's progress in this category
        progress = UserProgress.objects.get(
            user=request.user,
            category=attempt.quiz.category
        )
        
        # Calculate statistics
        correct_answers = answers.filter(is_correct=True).count()
        incorrect_answers = answers.filter(is_correct=False).count()
        
        context = {
            'attempt': attempt,
            'answers': answers,
            'feedback': feedback,
            'progress': progress,
            'correct_answers': correct_answers,
            'incorrect_answers': incorrect_answers,
            'passed': attempt.is_passed(),
        }
        
        return render(request, 'progress_tracking/quiz_results.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading results: {str(e)}')
        return redirect('progress_tracking_dashboard')

@login_required(login_url='/')
def user_analytics(request):
    """Detailed user analytics and progress"""
    try:
        # Get all user progress
        progress_records = UserProgress.objects.filter(user=request.user)
        
        # Get all completed attempts
        completed_attempts = QuizAttempt.objects.filter(
            user=request.user,
            completed=True
        ).order_by('-completed_at')
        
        # Calculate overall statistics
        total_quizzes = completed_attempts.count()
        total_xp = sum(p.xp_points for p in progress_records)
        overall_average = completed_attempts.aggregate(avg=Avg('score'))['avg'] or 0
        
        # Get performance over time (last 30 days)
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_attempts = completed_attempts.filter(completed_at__gte=thirty_days_ago)
        
        # Prepare chart data
        chart_data = []
        for attempt in recent_attempts[:10]:  # Last 10 attempts
            chart_data.append({
                'date': attempt.completed_at.strftime('%Y-%m-%d'),
                'score': attempt.score,
                'quiz_title': attempt.quiz.title[:20]
            })
        
        context = {
            'progress_records': progress_records,
            'completed_attempts': completed_attempts[:10],  # Recent 10
            'total_quizzes': total_quizzes,
            'total_xp': total_xp,
            'overall_average': round(overall_average, 2),
            'chart_data': json.dumps(chart_data),
        }
        
        return render(request, 'progress_tracking/analytics.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading analytics: {str(e)}')
        return redirect('progress_tracking_dashboard')

# ============== AI HELPER FUNCTIONS ==============

def generate_quiz_with_ai(category, difficulty, num_questions=10):
    """Generate quiz questions using Gemini AI"""
    try:
        api_key = 'AIzaSyDI8eZiJhgKlPfoxhSI_88u-6kEnrgOsyg'
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Create specialized prompts based on category
        category_prompts = {
            'programming': f"Create {num_questions} {difficulty} level programming fundamentals questions covering variables, data types, control structures, functions, and basic algorithms.",
            'dsa': f"Create {num_questions} {difficulty} level Data Structures and Algorithms questions covering arrays, linked lists, stacks, queues, trees, sorting, and searching algorithms.",
            'web_development': f"Create {num_questions} {difficulty} level web development questions covering HTML, CSS, JavaScript, frameworks, and web technologies.",
            'machine_learning': f"Create {num_questions} {difficulty} level machine learning questions covering supervised/unsupervised learning, algorithms, model evaluation, and data preprocessing.",
            'database': f"Create {num_questions} {difficulty} level database questions covering SQL, database design, normalization, indexing, and database management systems.",
            'system_design': f"Create {num_questions} {difficulty} level system design questions covering scalability, load balancing, caching, microservices, and distributed systems.",
            'cybersecurity': f"Create {num_questions} {difficulty} level cybersecurity questions covering network security, cryptography, authentication, and security best practices.",
            'mobile_development': f"Create {num_questions} {difficulty} level mobile development questions covering iOS/Android development, mobile UI/UX, and mobile technologies.",
            'devops': f"Create {num_questions} {difficulty} level DevOps questions covering CI/CD, containerization, cloud platforms, monitoring, and infrastructure as code.",
            'software_engineering': f"Create {num_questions} {difficulty} level software engineering questions covering SDLC, design patterns, testing, version control, and software architecture."
        }
        
        base_prompt = category_prompts.get(category.name, f"Create {num_questions} {difficulty} level {category.display_name} questions.")
        
        full_prompt = f"""
        {base_prompt}

        Return a JSON response with this exact structure:
        {{
            "title": "Brief descriptive title for this quiz",
            "description": "2-3 sentence description of what this quiz covers",
            "questions": [
                {{
                    "question": "Question text here",
                    "options": {{
                        "A": "Option A text",
                        "B": "Option B text", 
                        "C": "Option C text",
                        "D": "Option D text"
                    }},
                    "correct_answer": "A",
                    "explanation": "Brief explanation of why this answer is correct"
                }}
            ]
        }}

        Make sure:
        1. Questions are {difficulty} level appropriate
        2. All questions have exactly 4 options (A, B, C, D)
        3. Only one correct answer per question
        4. Explanations are clear and educational
        5. Questions test practical knowledge and understanding
        6. Return valid JSON only, no markdown or extra text
        """
        
        response = model.generate_content(full_prompt)
        
        # Clean and parse response
        response_text = response.text.strip()
        
        # Remove markdown formatting if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        quiz_data = json.loads(response_text)
        
        # Validate the response
        if not all(key in quiz_data for key in ['title', 'description', 'questions']):
            raise ValueError("Invalid quiz data structure")
        
        if len(quiz_data['questions']) != num_questions:
            raise ValueError("Incorrect number of questions generated")
        
        return quiz_data
        
    except Exception as e:
        print(f"Error generating quiz: {str(e)}")
        return None

def generate_ai_feedback(attempt):
    """Generate personalized feedback using AI"""
    try:
        api_key = 'AIzaSyDI8eZiJhgKlPfoxhSI_88u-6kEnrgOsyg'
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Get quiz performance data
        answers = QuizAnswer.objects.filter(attempt=attempt)
        correct_count = answers.filter(is_correct=True).count()
        total_count = answers.count()
        score_percentage = attempt.score
        
        # Analyze incorrect answers
        incorrect_questions = []
        for answer in answers.filter(is_correct=False):
            incorrect_questions.append({
                'question': answer.question.question_text[:100],
                'selected': answer.selected_answer,
                'correct': answer.question.correct_answer
            })
        
        prompt = f"""
        Analyze this quiz performance and provide personalized feedback:
        
        Quiz: {attempt.quiz.title} ({attempt.quiz.category.display_name})
        Difficulty: {attempt.quiz.difficulty}
        Score: {score_percentage}% ({correct_count}/{total_count})
        Time taken: {attempt.time_taken // 60} minutes
        
        Incorrect answers: {json.dumps(incorrect_questions[:5])}  # First 5 incorrect
        
        Provide feedback in this JSON format:
        {{
            "strengths": "2-3 sentences highlighting what the user did well",
            "weaknesses": "2-3 sentences identifying areas needing improvement", 
            "recommendations": "3-4 specific study recommendations based on incorrect answers",
            "next_steps": "Suggested next quizzes or topics to focus on for improvement"
        }}
        
        Make feedback:
        1. Encouraging and constructive
        2. Specific to the user's performance
        3. Actionable with clear next steps
        4. Focused on learning and improvement
        
        Return valid JSON only.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        feedback_data = json.loads(response_text)
        return feedback_data
        
    except Exception as e:
        print(f"Error generating feedback: {str(e)}")
        return None
