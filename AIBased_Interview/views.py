from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect  
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.db.models import Count, Avg, Q
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
from .models import QuizDomain, Quiz, Question, QuizSession, QuizAnswer
import re
import time
from django.views.decorators.http import require_POST
from django.core.mail import send_mail
from django.conf import settings
from django.middleware.csrf import get_token

def index(request):
    return render(request, 'index.html')

def test_csrf(request):
    """Test view to check CSRF token generation"""
    csrf_token = get_token(request)
    return HttpResponse(f"CSRF Token: {csrf_token}<br>Session ID: {request.session.session_key}")

@login_required(login_url='/') 
def homepage(request):
    return render(request, 'homepage.html')

@login_required(login_url='/') 
def prompt(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query = data.get('query', '').strip()
            
            if not query:
                return JsonResponse({'error': 'Query is required'}, status=400)
            
            # Import RAG service
            from .rag_service import rag_service
            
            # Process query using RAG
            result = rag_service.process_query(query, user=request.user)
            
            # Format similar documents for response
            similar_docs_formatted = []
            for doc in result['similar_documents']:
                similar_docs_formatted.append({
                    'question': doc['document']['question'],
                    'answer': doc['document']['answer'],
                    'similarity_score': round(doc['similarity_score'], 3),
                    'rank': doc['rank']
                })
            
            response_data = {
                'query': result['query'],
                'response': result['response'],
                'similar_documents': similar_docs_formatted,
                'processing_time': round(result['processing_time'], 3),
                'num_documents_found': len(similar_docs_formatted)
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return render(request, 'Prompt.html')

@login_required(login_url='/') 
def about(request):
    return render(request, 'About Us.html')

@login_required(login_url='/')
def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        full_message = f"""
        New contact form submission:

        Name: {name}
        Email: {email}
        Subject: {subject}

        Message:
        {message}
        """

        send_mail(
            subject=f"Contact Form: {subject}",
            message=full_message,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=['emanmunir740@gmail.com'],
            fail_silently=False,
        )

        messages.success(request, "Your message has been sent successfully!")
        return redirect('contact')

    return render(request, 'Contact.html')


@login_required(login_url='/') 
def cv_based(request):
    if request.method == 'POST':
        try:
            if 'cv_file' not in request.FILES:
                return JsonResponse({'error': 'No file uploaded'}, status=400)
            
            cv_file = request.FILES['cv_file']
            num_questions = int(request.POST.get('num_questions'))
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
                questions = generate_questions_with_gemini(cv_text = cv_text, num_questions = num_questions)
                
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
    
    return render(request, 'CV_Based.html')

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
    
    return render(request, 'Open_AI_Prompt.html')

@login_required(login_url='/') 
def main_prompt(request): 
    return render(request, 'Main_Prompt.html')

@login_required(login_url='/') 
def generic_prompt(request):
    return HttpResponse("This is the generic prompt page.")

@csrf_protect 
def login(request):
    print(f"Login view accessed: Method={request.method}, GET params={request.GET}, POST params={request.POST}")
    print(f"CSRF Token: {request.META.get('CSRF_COOKIE', 'Not found')}")
    print(f"Session ID: {request.session.session_key}")
    
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

@csrf_protect  
def signup(request):
    print(f"Signup view accessed: Method={request.method}, GET params={request.GET}, POST params={request.POST}")
    print(f"CSRF Token: {request.META.get('CSRF_COOKIE', 'Not found')}")
    print(f"Session ID: {request.session.session_key}")
    
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
@require_POST
def chat_support(request):
    """Handle chat support messages with AI assistance"""
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Generate AI response using Gemini
        ai_response = generate_chat_response(message)
        
        return JsonResponse({
            'response': ai_response,
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def generate_chat_response(user_message: str) -> str:
    """Generate AI response for chat support using Gemini API"""
    api_key = 'AIzaSyDI8eZiJhgKlPfoxhSI_88u-6kEnrgOsyg'
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Context about the website and its features
    context = """
    You are a helpful AI assistant for PrepareInterview, an AI-powered interview preparation platform. 
    Here's what you need to know about our website:

    WEBSITE FEATURES:
    1. CV-Based Interview Questions: Users can upload their CV (.docx files) and get personalized interview questions
    2. OpenAI Code Generation: Users can generate code in various programming languages (Python, Java, JavaScript, C, C++, Go, Ruby)
    3. Confidence Checker: Users can upload their photo to get AI analysis of their confidence level with personalized advice
    4. Quiz System: Comprehensive quiz system with 15+ domains (Python, JavaScript, Java, React, Django, Data Science, ML, etc.)
       - Users can select difficulty levels (Easy, Medium, Hard)
       - Timed quizzes with 5-25 questions
       - Detailed results with explanations
       - Performance tracking and history

    HOW TO USE:
    - Sign up/Login with email and password
    - Access homepage after login
    - Navigate to different features from the homepage
    - Take quizzes by selecting domain, difficulty, and question count
    - Upload CV for personalized interview questions
    - Use code generation for programming practice
    - Check confidence with photo analysis

    TECHNICAL DETAILS:
    - Built with Django and AI integration
    - Uses Gemini AI for content generation
    - Professional cyber-themed UI design
    - Responsive design for all devices
    - Secure user authentication system

    Please help users with questions about:
    - How to use different features
    - Troubleshooting issues
    - Understanding quiz results
    - Getting the most out of the platform
    - Technical requirements (file formats, etc.)
    
    Instructions:
    - Always provide clear, concise answers based on the context.
    - If the question is not covered in the context, suggest contacting support.
    - Use a friendly and professional tone.
    - If the user asks about a specific feature, provide detailed information.
    - If the user asks a out of the platform question, provide a helpful response but dont answer that question, only stick to the platform related questions.

    Be friendly, helpful, and professional. If you don't know something specific, be honest and suggest they contact support.
    """
    
    full_prompt = f"{context}\n\nUser Question: {user_message}\n\nProvide a helpful, concise response:"
    
    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return "I'm sorry, I'm having trouble connecting right now. Please try again in a moment or contact our support team for immediate assistance."

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

# ===================== QUIZ SYSTEM VIEWS =====================

@login_required(login_url='/')
def quiz_dashboard(request):
    """Quiz dashboard showing user's quiz statistics and available domains"""
    user = request.user
    
    # User's quiz statistics
    total_sessions = QuizSession.objects.filter(user=user).count()
    completed_sessions = QuizSession.objects.filter(user=user, status='completed').count()
    avg_score = QuizSession.objects.filter(user=user, status='completed').aggregate(
        avg_score=Avg('percentage')
    )['avg_score'] or 0
    
    # Recent quiz sessions
    recent_sessions = QuizSession.objects.filter(user=user).order_by('-started_at')[:5]
    
    # Domain-wise statistics
    domain_stats = QuizSession.objects.filter(user=user, status='completed').values(
        'quiz__domain__display_name', 'quiz__domain__icon'
    ).annotate(
        total_quizzes=Count('id'),
        avg_score=Avg('percentage')
    ).order_by('-avg_score')
    
    # Available domains
    domains = QuizDomain.objects.filter(is_active=True)
    
    # Get the latest 10 quiz sessions for the logged-in user
    sessions = QuizSession.objects.filter(user=request.user).order_by('-started_at')[:10]
    sessions = list(reversed(sessions))  # Reverse to show oldest first (1 to 10)

    # Prepare data for the chart
    quiz_labels = [f"Quiz {i+1}" for i in range(len(sessions))]
    percentages = [session.percentage for session in sessions]
    
    context = {
        'total_sessions': total_sessions,
        'completed_sessions': completed_sessions,
        'avg_score': round(avg_score, 1),
        'recent_sessions': recent_sessions,
        'domain_stats': domain_stats,
        'domains': domains,
        'quiz_labels': quiz_labels,
        'percentages': percentages,
    }
    return render(request, 'quiz/dashboard.html', context)

@login_required(login_url='/')
def quiz_domain_selection(request):
    """Display available quiz domains for selection"""
    domains = QuizDomain.objects.filter(is_active=True)
    
    # Add quiz count for each domain
    for domain in domains:
        domain.quiz_count = domain.quizzes.count()
        domain.user_attempts = QuizSession.objects.filter(
            user=request.user, 
            quiz__domain=domain
        ).count()
    
    return render(request, 'quiz/domain_selection.html', {'domains': domains})

@login_required(login_url='/')
def quiz_setup(request, domain_id):
    """Quiz setup page where user selects difficulty and number of questions"""
    domain = get_object_or_404(QuizDomain, id=domain_id, is_active=True)
    
    if request.method == 'POST':
        difficulty = request.POST.get('difficulty', 'medium')
        num_questions = int(request.POST.get('num_questions', 10))
        time_limit = int(request.POST.get('time_limit', 15))
        
        # Validate inputs
        if difficulty not in ['easy', 'medium', 'hard']:
            difficulty = 'medium'
        if num_questions < 5 or num_questions > 50:
            num_questions = 10
        if time_limit < 5 or time_limit > 60:
            time_limit = 15
        
        # Generate quiz using AI
        try:
            quiz = generate_ai_quiz(request.user, domain, difficulty, num_questions, time_limit)
            return redirect('quiz_start', quiz_id=quiz.id)
        except Exception as e:
            messages.error(request, f'Error generating quiz: {str(e)}')
            return redirect('quiz_domain_selection')
    
    return render(request, 'quiz/setup.html', {'domain': domain})

@login_required(login_url='/')
def quiz_start(request, quiz_id):
    """Start a new quiz session"""
    quiz = get_object_or_404(Quiz, id=quiz_id)
    
    # Check if user already has an active session for this quiz
    active_session = QuizSession.objects.filter(
        user=request.user, 
        quiz=quiz, 
        status='in_progress'
    ).first()
    
    if active_session:
        return redirect('quiz_take', session_id=active_session.id)
    
    # Create new quiz session
    session = QuizSession.objects.create(
        user=request.user,
        quiz=quiz,
        total_questions=quiz.total_questions,
        status='in_progress'
    )
    
    return redirect('quiz_take', session_id=session.id)

@login_required(login_url='/')
def quiz_take(request, session_id):
    """Take the quiz - display questions and handle answers"""
    session = get_object_or_404(QuizSession, id=session_id, user=request.user)
    
    if session.status != 'in_progress':
        return redirect('quiz_result', session_id=session.id)
    
    questions = session.quiz.questions.all()
    answered_questions = QuizAnswer.objects.filter(session=session).values_list('question_id', flat=True)
    
    # Find next unanswered question
    next_question = None
    for question in questions:
        if question.id not in answered_questions:
            next_question = question
            break
    
    if not next_question:
        # All questions answered, complete the quiz
        return redirect('quiz_complete', session_id=session.id)
    
    # Calculate progress
    progress = (len(answered_questions) / session.total_questions) * 100
    question_number = len(answered_questions) + 1
    
    if request.method == 'POST':
        selected_answer = request.POST.get('answer')
        question_time = int(request.POST.get('time_taken', 0))
        
        if selected_answer:
            # Save the answer
            quiz_answer = QuizAnswer.objects.create(
                session=session,
                question=next_question,
                selected_answer=selected_answer,
                time_taken=question_time
            )
            
            # Update session statistics
            if quiz_answer.is_correct:
                session.correct_answers += 1
            else:
                session.wrong_answers += 1
            session.save()
            
            # Return success without revealing correct answer or feedback
            return JsonResponse({
                'success': True,
                'message': 'Answer submitted successfully'
            })
    
    context = {
        'session': session,
        'question': next_question,
        'question_number': question_number,
        'total_questions': session.total_questions,
        'progress': round(progress, 1),
        'time_limit_minutes': session.quiz.time_limit,
    }
    return render(request, 'quiz/take.html', context)

@login_required(login_url='/')
def quiz_complete(request, session_id):
    """Complete the quiz and calculate final results"""
    session = get_object_or_404(QuizSession, id=session_id, user=request.user)
    quiz_answers = QuizAnswer.objects.filter(session=session)
    total_time = sum(answer.time_taken for answer in quiz_answers)
    
    if session.status == 'completed':
        return redirect('quiz_result', session_id=session.id)
    # Calculate total time taken
    # time_taken = int(request.POST.get('total_time', 0))
    session.time_taken = total_time
    session.status = 'completed'
    session.completed_at = timezone.now()
    session.save()
    
    
    return redirect('quiz_result', session_id=session.id)

@login_required(login_url='/')
def quiz_result(request, session_id):
    """Display quiz results and detailed analysis"""
    session = get_object_or_404(QuizSession, id=session_id, user=request.user)
    
    # Get all answers with questions
    answers = QuizAnswer.objects.filter(session=session).select_related('question')
    correct_answers = answers.filter(is_correct=True)
    wrong_answers = answers.filter(is_correct=False)
    
    # Calculate time statistics
    avg_time_per_question = session.time_taken / session.total_questions if session.total_questions > 0 else 0
    
    # Domain performance comparison
    domain_avg = QuizSession.objects.filter(
        quiz__domain=session.quiz.domain,
        status='completed'
    ).aggregate(avg_score=Avg('percentage'))['avg_score'] or 0
    
    context = {
        'session': session,
        'answers': answers,
        'correct_answers': correct_answers,
        'wrong_answers': wrong_answers,
        'avg_time_per_question': round(avg_time_per_question, 1),
        'domain_avg': round(domain_avg, 1),
        'grade': session.get_grade(),
    }
    return render(request, 'quiz/result.html', context)

@login_required(login_url='/')
def quiz_history(request):
    """Display user's quiz history with filtering and pagination"""
    sessions = QuizSession.objects.filter(user=request.user).select_related('quiz__domain')
    
    # Filtering
    domain_filter = request.GET.get('domain')
    status_filter = request.GET.get('status')
    
    if domain_filter:
        sessions = sessions.filter(quiz__domain__name=domain_filter)
    if status_filter:
        sessions = sessions.filter(status=status_filter)
    
    # Pagination
    paginator = Paginator(sessions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Available domains for filter
    domains = QuizDomain.objects.filter(is_active=True)
    
    context = {
        'page_obj': page_obj,
        'domains': domains,
        'current_domain': domain_filter,
        'current_status': status_filter,
    }
    return render(request, 'quiz/history.html', context)

def generate_ai_quiz(user, domain, difficulty, num_questions, time_limit):
    """Generate quiz questions using AI"""
    api_key = 'AIzaSyDI8eZiJhgKlPfoxhSI_88u-6kEnrgOsyg'
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Create quiz object
    quiz = Quiz.objects.create(
        name=f"{domain.display_name} Quiz - {difficulty.title()}",
        description=f"AI-generated {difficulty} level quiz on {domain.display_name}",
        domain=domain,
        difficulty=difficulty,
        total_questions=num_questions,
        time_limit=time_limit,
        created_by=user,
        is_ai_generated=True
    )
    
    # Generate questions prompt
    prompt = f"""
    Generate {num_questions} multiple choice questions for {domain.display_name} at {difficulty} difficulty level.
    
    Format each question exactly as follows:
    Q: [Question text]
    A) [Option A]
    B) [Option B] 
    C) [Option C]
    D) [Option D]
    ANSWER: [A/B/C/D]
    EXPLANATION: [Brief explanation of the correct answer]
    ---
    
    Requirements:
    - Questions should be practical and relevant to {domain.display_name}
    - {difficulty.title()} difficulty level appropriate
    - Clear, unambiguous questions
    - Realistic, plausible wrong answers
    - Brief but informative explanations
    """
    
    try:
        response = model.generate_content(prompt)
        questions_text = response.text
        
        # Parse the generated questions
        parse_and_save_questions(quiz, questions_text)
        
        return quiz
        
    except Exception as e:
        quiz.delete()  # Clean up if question generation fails
        raise Exception(f"Failed to generate questions: {str(e)}")

def parse_and_save_questions(quiz, questions_text):
    """Parse AI-generated questions and save to database"""
    questions = questions_text.split('---')
    
    for i, question_block in enumerate(questions):
        question_block = question_block.strip()
        if not question_block:
            continue
            
        try:
            # Extract question components using regex
            question_match = re.search(r'Q:\s*(.+?)(?=A\))', question_block, re.DOTALL)
            option_a_match = re.search(r'A\)\s*(.+?)(?=B\))', question_block, re.DOTALL)
            option_b_match = re.search(r'B\)\s*(.+?)(?=C\))', question_block, re.DOTALL)
            option_c_match = re.search(r'C\)\s*(.+?)(?=D\))', question_block, re.DOTALL)
            option_d_match = re.search(r'D\)\s*(.+?)(?=ANSWER:)', question_block, re.DOTALL)
            answer_match = re.search(r'ANSWER:\s*([ABCD])', question_block)
            explanation_match = re.search(r'EXPLANATION:\s*(.+?)$', question_block, re.DOTALL)
            
            if all([question_match, option_a_match, option_b_match, option_c_match, option_d_match, answer_match]):
                Question.objects.create(
                    quiz=quiz,
                    question_text=question_match.group(1).strip(),
                    option_a=option_a_match.group(1).strip(),
                    option_b=option_b_match.group(1).strip(),
                    option_c=option_c_match.group(1).strip(),
                    option_d=option_d_match.group(1).strip(),
                    correct_answer=answer_match.group(1),
                    explanation=explanation_match.group(1).strip() if explanation_match else "",
                    difficulty=quiz.difficulty
                )
        except Exception as e:
            print(f"Error parsing question {i+1}: {str(e)}")
            continue
    
    # Verify we have enough questions
    actual_questions = quiz.questions.count()
    if actual_questions == 0:
        raise Exception("No valid questions were generated")
    
    # Update quiz with actual question count
    quiz.total_questions = actual_questions
    quiz.save()