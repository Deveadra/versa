import os
import pickle
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
TOKEN_PATH = "token_gmail.pickle"
_service = None


def init_gmail_service():
    """Authenticate and build Gmail service."""
    global _service
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)
    _service = build("gmail", "v1", credentials=creds)
    return _service


def get_unread_emails(n=5):
    """Fetch unread emails from Gmail (fallback to mock if unavailable)."""
    global _service
    if not _service:
        return "[Mock] No Gmail service available."

    results = _service.users().messages().list(userId="me", labelIds=["INBOX"], q="is:unread", maxResults=n).execute()
    messages = results.get("messages", [])
    if not messages:
        return "No unread emails."

    output = []
    for m in messages:
        msg = _service.users().messages().get(userId="me", id=m["id"]).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        subject = headers.get("Subject", "(no subject)")
        sender = headers.get("From", "(unknown sender)")
        output.append(f"From: {sender} | Subject: {subject}")
    return "\n".join(output)


def send_email(recipient, subject, body):
    """Send an email via Gmail (fallback to mock if unavailable)."""
    from email.mime.text import MIMEText
    import base64

    global _service
    if not _service:
        return f"[Mock] Would send email to {recipient} with subject '{subject}'."

    message = MIMEText(body)
    message["to"] = recipient
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    message_body = {"raw": raw}

    sent = _service.users().messages().send(userId="me", body=message_body).execute()
    return f"Email sent to {recipient} with subject '{subject}' (id: {sent['id']})."