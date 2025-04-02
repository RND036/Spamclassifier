import string
from django.shortcuts import render
import joblib
import sklearn

model = joblib.load("spamclassifiermodel.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def process_data(text):
  
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())

def home(request):
   
    result = ""
    
    if request.method == "POST":
        email_text = request.POST.get("email_text", "")
        cleaned_email = process_data(email_text)
        
        if cleaned_email:  # Only process if the email is not empty
            vectorized_email = vectorizer.transform([cleaned_email])
            prediction = model.predict(vectorized_email)
            result = "This email is Spam" if prediction[0] == 1 else "This email is Not Spam"

    return render(request, "home.html", {"result": result})
