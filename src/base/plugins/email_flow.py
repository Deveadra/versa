import random
import re

from base.apps import email_prompts as prompts
from base.core.context import clear_context, get_context, set_context
from base.core.profile import get_pref
from base.plugins.gmail import (
    get_unread_emails,
    send_email,  # your real Gmail module
)


def parse_quick_email(text: str):
    """
    Try to parse a one-shot email command (recipient, subject, body).
    Returns dict or None.
    """
    recipient, subject, body = None, None, None

    # "Send email to alice@example.com about project: let's meet"
    match_to = re.search(r"to ([\w\.-]+@[\w\.-]+)", text, re.IGNORECASE)
    if match_to:
        recipient = match_to.group(1)

    if "subject" in text.lower() and "body" in text.lower():
        # "subject X body Y"
        parts = text.split("subject", 1)[1]
        if "body" in parts:
            subject, body = parts.split("body", 1)
            subject, body = subject.strip(), body.strip()
    elif "about" in text.lower() and ":" in text:
        # "about X: Y"
        about_part = text.split("about", 1)[1]
        subject, body = about_part.split(":", 1)
        subject, body = subject.strip(), body.strip()

    if recipient and subject and body:
        return {"recipient": recipient, "subject": subject, "body": body, "confirm": True}
    return None


def continue_email_flow(text: str):
    """
    Continue filling out email draft fields until ready to send.
    """
    draft = get_context("email_draft")
    if not draft:
        return None, None

    if draft["recipient"] is None:
        draft["recipient"] = text.strip()
        set_context("email_draft", draft)
        return None, random.choice(prompts.ASK_SUBJECT_VARIANTS)

    if draft["subject"] is None:
        draft["subject"] = text.strip()
        set_context("email_draft", draft)
        return None, random.choice(prompts.ASK_BODY_VARIANTS)

    if draft["body"] is None:
        draft["body"] = text.strip()
        draft["confirm"] = True
        set_context("email_draft", draft)
        confirm_prompt = random.choice(prompts.ASK_CONFIRM_VARIANTS).format(
            recipient=draft["recipient"], subject=draft["subject"], body=draft["body"]
        )
        return None, confirm_prompt

    if draft.get("confirm"):
        if text.lower() in ["yes", "send it", "send", "confirm", "do it"]:
            try:
                send_email(draft["recipient"], draft["subject"], draft["body"])
                clear_context("email_draft")
                return "Email sent.", random.choice(prompts.CONFIRM_SEND_VARIANTS)
            except Exception as e:
                clear_context("email_draft")
                return f"Failed to send: {e}", "Sorry, I couldn’t send the email."
        else:
            clear_context("email_draft")
            return "Cancelled.", random.choice(prompts.CANCEL_SEND_VARIANTS)

    return None, None


def handle_email_command(text):
    """
    Handle email-related requests.
    """
    default_email = get_pref("default_email", None)

    if "unread" in text.lower():
        unread = get_unread_emails(n=5)  # account=default_email)
        if not unread:
            return "No unread emails.", "You have no unread messages."

        first = unread[0]
        if isinstance(first, dict):
            sender = str(first.get("from", "someone"))
            subj = str(first.get("subject", "(no subject)"))
        else:
            # fallbacks if your gmail helper returns strings/objects
            sender = str(first)
            subj = ""

        spoken = f"You have {len(unread)} unread. Latest from {sender}" + (
            f" about {subj}." if subj else "."
        )
        return None, spoken

    if "send" in text.lower():
        # Use default email account if none specified
        return None, f"Okay, I’ll send it from {default_email or 'your account'}."

    return None, None


# def handle_email_command(text: str):
#     """
#     Orchestrator: route text to quick parse, flow continuation, or new draft.
#     """
#     text_lower = text.lower()

#     # Send existing draft
#     if text_lower in ["send it", "send the email"]:
#         draft = get_context("email_draft")
#         if draft:
#             return continue_email_flow("yes")
#         return "No draft to send.", "You don’t have any draft ready."

#     # Existing draft in progress → continue
#     draft = get_context("email_draft")
#     if draft:
#         return continue_email_flow(text)

#     # Try parsing one-shot command
#     quick = parse_quick_email(text)
#     if quick:
#         set_context("email_draft", quick)
#         return None, random.choice(prompts.ASK_CONFIRM_VARIANTS).format(
#             recipient=quick["recipient"], subject=quick["subject"], body=quick["body"]
#         )

#     # New draft
#     if "email" in text_lower or "compose" in text_lower:
#         draft = {"recipient": None, "subject": None, "body": None, "confirm": False}
#         set_context("email_draft", draft)
#         return "Who do you want to send it to?", random.choice(prompts.ASK_RECIPIENT_VARIANTS)

#     return None, None
