# Email Spam Classification Using Machine Learning

Beginner-friendly Python mini-project using TF-IDF and Multinomial Naive Bayes to classify messages as `spam` or `ham`.

## Files
- `emails.csv`: labeled sample messages (`label`, `text` columns)
- `spam_classifier.py`: terminal classifier and evaluation metrics
- `app.py`: Streamlit web interface
- `Project_Report.docx`: project report
- `Project_Presentation.pptx`: 12-slide presentation
- `START_HERE_MARATHI.txt`: Marathi quick start

## Run on Windows / VS Code
Open this folder in VS Code and choose Terminal > New Terminal.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python spam_classifier.py
```

Run the web interface:
```powershell
streamlit run app.py
```

The included dataset is small and intended only for learning. Results can vary and are not representative of real-world performance. For a formal project, use a larger documented dataset with suitable licensing and report the actual results from your own run. Do not rely on this demo for real email security decisions.
