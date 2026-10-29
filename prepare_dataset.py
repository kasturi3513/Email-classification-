
from pathlib import Path
from email import policy
from email.parser import BytesParser
import pandas as pd

ROOT = Path(__file__).parent
OUTPUT = ROOT / "emails.csv"

def get_label(file_path):
    parts = [part.lower() for part in file_path.relative_to(ROOT).parts[:-1]]

    for part in parts:
        if "easy_ham" in part or part in ("ham", "easyham"):
            return "ham"
        if "spam" in part and "ham" not in part:
            return "spam"

    return None

def extract_text(message):
    subject = str(message.get("subject", "") or "")
    body_parts = []

    parts = message.walk() if message.is_multipart() else [message]

    for part in parts:
        if part.get_content_type() != "text/plain":
            continue
        if part.get_content_disposition() == "attachment":
            continue

        try:
            content = part.get_content()
            if isinstance(content, str):
                body_parts.append(content)
        except Exception:
            continue

    return (subject + "\n" + "\n".join(body_parts)).strip()

rows = []
seen = set()
skipped = 0

for file_path in ROOT.rglob("*"):
    if not file_path.is_file() or file_path == OUTPUT:
        continue

    label = get_label(file_path)
    if label is None:
        continue

    try:
        with file_path.open("rb") as file:
            message = BytesParser(policy=policy.default).parse(file)

        text = extract_text(message)
        if len(text) < 20:
            skipped += 1
            continue

        key = text.casefold()
        if key in seen:
            continue

        seen.add(key)
        rows.append({"label": label, "text": text})

    except Exception:
        skipped += 1

data = pd.DataFrame(rows, columns=["label", "text"])
data.to_csv(OUTPUT, index=False, encoding="utf-8")

print("Dataset saved to:", OUTPUT)
print("Total emails:", len(data))
print("\nLabel counts:")
print(data["label"].value_counts().to_string())
print("\nSkipped files:", skipped)

if data.empty or data["label"].nunique() < 2:
    print("\nWARNING: Both ham and spam labels were not found.")
    print("Check that the extracted folders are inside the project folder.")
