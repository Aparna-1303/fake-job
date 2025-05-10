from django.shortcuts import render,redirect
import joblib
from django.conf import settings
import os
from .models import Userinfo
# Create your views here.
def home(request):
    return render(request,"home.html")
def login(request):
    if request.method=='POST':
        username=request.POST['username']
        password=request.POST['password']
        user=Userinfo.objects.filter(username=username,password=password).exists()
        if user:
            return redirect("response")
        else:
            return redirect("login")
    return render(request,'login.html')
def signup(request):
    if request.method=='POST':
        username=request.POST['username']
        email=request.POST['email']
        password=request.POST['password']
        Userinfo.objects.create(username=username,email=email,password=password)

    return render(request,"signup.html")

# Load the model and vectorizer once
MODEL_PATH = os.path.join(settings.BASE_DIR, 'static', 'Model.pkl')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'static', 'Vectorizer.pkl')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def response(request):
    prediction = None

    if request.method == 'POST':
        title = request.POST.get('title', '')
        location = request.POST.get('location', '')
        description = request.POST.get('description', '')
        requirements = request.POST.get('requirements', '')
        telecommuting = int(request.POST.get('telecommuting', 0))
        has_company_logo = int(request.POST.get('has_company_logo', 0))
        has_questions = int(request.POST.get('has_questions', 0))

        combined_text = f"{title} {location} {description} {requirements}"
        text_features = vectorizer.transform([combined_text])
        numeric_features = [[telecommuting, has_company_logo, has_questions]]

        import scipy
        final_features = scipy.sparse.hstack([text_features, numeric_features])

        pred = model.predict(final_features)[0]
        prediction = "Fake" if pred == 1 else "Real"

    return render(request, 'response.html', {'prediction': prediction})