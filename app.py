SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_USER = 'user_mail@gmail.com'          # replace with your FROM address
EMAIL_PASS = 'password'       # use a Gmail app-specific password
EMAIL_TO = 'TO_mail@gmail.com'            # or use a different mailbox for receiving


import os
import io
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import matplotlib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file, session
from werkzeug.utils import secure_filename
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from wordcloud import WordCloud

matplotlib.use('Agg')  # Use non-GUI backend for plots

# === Flask Configuration ===
app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a secure value
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# === Load & Train Model at Startup ===
df = pd.read_csv('preprocessed_reviews.csv')
X, y = df['cleaned_review'], df['sentiment']
vectorizer = TfidfVectorizer(max_features=3000)
model = LogisticRegression(max_iter=1000)
pipeline = make_pipeline(vectorizer, model)
pipeline.fit(X, y)

explainer = LimeTextExplainer(class_names=['negative', 'positive'])
last_explanation_html = ""

# === Home: Single + Batch Prediction ===
@app.route('/', methods=['GET', 'POST'])
def index():
    global last_explanation_html

    prediction = None
    probability = None
    explanation_html = None
    review_text = ''
    batch_results = None
    history = session.get('history', [])

    if request.method == 'POST':
        print("---- DEBUG ----")
        print("FORM:", request.form)

        review_text = request.form.get('review', '').strip()

    # === SINGLE PREDICTION ===
    if review_text:
        print("RUNNING SINGLE PREDICTION")

        try:
            pred = pipeline.predict([review_text])[0]
            prob = pipeline.predict_proba([review_text])[0]

            prediction = pred.lower()
            probability = f"Probabilities — Positive: {prob[1]:.2%} | Negative: {prob[0]:.2%}"

            try:
                exp = explainer.explain_instance(
                    review_text,
                    pipeline.predict_proba,
                    num_features=10
                )
                explanation_html = exp.as_html()
                last_explanation_html = explanation_html
            except Exception as e:
                print("LIME ERROR:", e)
                explanation_html = "<p style='color:red;'>LIME failed</p>"

            # Save history
            record = {
                'text': review_text[:100] + '...' if len(review_text) > 100 else review_text,
                'sentiment': pred.capitalize(),
                'prob_pos': f"{prob[1]:.2%}",
                'prob_neg': f"{prob[0]:.2%}"
            }
            history.append(record)
            session['history'] = history[-5:]

        except Exception as e:
            print("ERROR:", e)
            prediction = "Prediction failed"

    # === BATCH PREDICTION ===
    elif 'reviews_file' in request.files:
        print("RUNNING BATCH")

        uploaded_file = request.files.get('reviews_file')

        if uploaded_file and uploaded_file.filename.endswith('.csv'):
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            try:
                batch_df = pd.read_csv(filepath)

                if 'review' not in batch_df.columns:
                    batch_results = "CSV must have a 'review' column."
                else:
                    batch_df['predicted_sentiment'] = pipeline.predict(batch_df['review'])
                    batch_df['prob_positive'] = pipeline.predict_proba(batch_df['review'])[:, 1]
                    batch_df['prob_negative'] = pipeline.predict_proba(batch_df['review'])[:, 0]

                    batch_results = batch_df.head(20).to_html(
                        classes='table table-striped',
                        index=False
                    )

            except Exception as e:
                batch_results = f"Error: {e}"

        else:
            batch_results = "Upload valid CSV"

    return render_template(
        'index.html',
        prediction=prediction,
        probability=probability,
        explanation_html=explanation_html,
        review_text=review_text,
        batch_results=batch_results,
        history=history
    )

# === Download LIME Explanation as HTML File ===
@app.route('/download_explanation')
def download_explanation():
    global last_explanation_html
    if not last_explanation_html:
        return "No explanation to download. Please analyze a review first.", 400
    return send_file(
        io.BytesIO(last_explanation_html.encode('utf-8')),
        mimetype='text/html',
        as_attachment=True,
        download_name='lime_explanation.html'
    )

# === Word Cloud Page ===
@app.route('/wordcloud')
def wordcloud():
    positive_text = ' '.join(df[df['sentiment'] == 'positive']['cleaned_review'])
    negative_text = ' '.join(df[df['sentiment'] == 'negative']['cleaned_review'])

    def wc_img(text):
        wc = WordCloud(width=600, height=400, background_color='white').generate(text)
        img = io.BytesIO()
        plt.figure(figsize=(7, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    pos_img = wc_img(positive_text)
    neg_img = wc_img(negative_text)

    return render_template('wordcloud.html',
                           pos_img=pos_img,
                           neg_img=neg_img)

# === Contact Page: GET + POST (Form Submission) ===

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    success_message = None
    error_message = None

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        message = request.form.get('message', '').strip()

        if not name or not email or not message:
            error_message = "All fields are required."
        elif '@' not in email:
            error_message = "Enter a valid email address."
        else:
            # === Send Email via SMTP ===
            try:
                msg = MIMEMultipart()
                msg['From'] = EMAIL_USER
                msg['To'] = EMAIL_TO
                msg['Subject'] = f"New Contact Form Submission from {name}"

                body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
                msg.attach(MIMEText(body, 'plain'))

                server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASS)
                server.send_message(msg)
                server.quit()

                success_message = "✅ Thank you! Your message has been sent."
            except Exception as e:
                error_message = f"❌ Failed to send message: {e}"

    return render_template('contact.html',
                           success_message=success_message,
                           error_message=error_message)


# === About Page ===
@app.route('/about')
def about():
    return render_template('about.html')

# === Model Performance Page ===
@app.route('/performance')
def performance():
    y_pred = pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred, labels=['positive', 'negative'])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['positive', 'negative'],
                yticklabels=['positive', 'negative'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template('performance.html',
                           accuracy=round(accuracy, 4),
                           report=report,
                           plot_url=plot_url)

# === File Size Error Handler ===
@app.errorhandler(413)
def too_large(e):
    return "❌ File too large. Max upload: 100 MB.", 413



@app.route('/export_pdf')
def export_pdf():
    history = session.get('history', [])
    pdf_file = io.BytesIO()
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Sentiment Analysis Report", styles['Title'])]
    elements.append(Spacer(1, 12))

    data = [["Input Preview", "Sentiment", "Positive %", "Negative %"]]
    for item in history:
        data.append([item['text'], item['sentiment'], item['prob_pos'], item['prob_neg']])

    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(table)
    doc.build(elements)
    pdf_file.seek(0)
    return send_file(
        pdf_file,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='sentiment_report.pdf'
    )


# === Run Flask App ===
if __name__ == '__main__':
    app.run(debug=True)
