import random
import re
from base.apps import email_prompts as prompts
from base.core.context import set_context, get_context, clear_context


# Conversation state for email composition
email_pending = {"recipient": None, "subject": None, "body": None, "confirm": False}


def handle_email_command(text):
    """
    Main entry point for email plugin.
    """
    text_lower = text.lower()

    # Check if user wants to send a draft
    if "send it" in text_lower or "send the email" in text_lower:
        draft = get_context("email_draft")
        if draft:
            # simulate sending
            clear_context("email_draft")
            return "Email sent.", "The email has been sent."
        else:
            return "No draft to send.", "You don’t have any email draft ready."

    # Start composing
    if "email" in text_lower or "compose" in text_lower:
        draft = {"recipient": None, "subject": None, "body": None}
        set_context("email_draft", draft)
        return "Who do you want to send it to?", "Who should I address the email to?"

    # If we’re in the middle of a draft, capture missing fields
    draft = get_context("email_draft")
    if draft:
        if draft["recipient"] is None:
            draft["recipient"] = text
            set_context("email_draft", draft)
            return "What’s the subject?", "What is the subject of the email?"
        elif draft["subject"] is None:
            draft["subject"] = text
            set_context("email_draft", draft)
            return "What’s the body?", "What would you like the email to say?"
        elif draft["body"] is None:
            draft["body"] = text
            set_context("email_draft", draft)
            return "Say 'send it' when you’re ready.", "Draft complete. Say 'send it' to send."
    
    return None, None

# from base.core.stylizer import stylize_response


# # For mock/demo purposes
# MOCK_UNREAD = [
#     {"from": "alice@example.com", "subject": "Meeting tomorrow"},
#     {"from": "bob@example.com", "subject": "Invoice update"}
# ]

def has_pending():
    return any(v is None for k, v in email_pending.items() if k != "confirm") and any(v is not None for v in email_pending.values()) or email_pending["confirm"]


def is_email_command(text: str) -> bool:
    return "email" in text.lower() and ("send" in text.lower() or "compose" in text.lower())


# def handle(text, personality=None, mode="default"):
#     """
#     Handle email commands: check unread emails, compose/send, etc.
#     """

#     # Simplified: always return unread count for now
#     unread_count = len(MOCK_UNREAD)

#     if unread_count == 0:
#         if personality:
#             return stylize_response(personality, mode, "email_empty", {})
#         return "You have no unread emails."

#     latest = MOCK_UNREAD[0]
#     data = {"count": unread_count, "from": latest["from"], "subject": latest["subject"]}

#     if personality:
#         return stylize_response(personality, mode, "email", data)

#     return f"You have {unread_count} unread emails. Latest from {latest['from']} about {latest['subject']}."

def handle(text: str, active_plugins):
    """
    Handle email flow.
    Returns: (reply_to_console, spoken_response)
    """
    global email_pending

    # Follow-up flow
    if has_pending():
        if email_pending["subject"] is None:
            email_pending["subject"] = text.strip()
            return None, random.choice(prompts.ASK_BODY_VARIANTS)

        elif email_pending["body"] is None:
            email_pending["body"] = text.strip()
            confirm_prompt = random.choice(prompts.ASK_CONFIRM_VARIANTS).format(
                subject=email_pending["subject"], body=email_pending["body"]
            )
            email_pending["confirm"] = True
            return None, confirm_prompt

        elif email_pending["confirm"]:
            if text.lower() in ["yes", "send", "confirm", "do it"]:
                reply = active_plugins["send_email"](
                    email_pending["recipient"], email_pending["subject"], email_pending["body"]
                )
                email_pending = {"recipient": None, "subject": None, "body": None, "confirm": False}
                return reply, random.choice(prompts.CONFIRM_SEND_VARIANTS)
            else:
                email_pending = {"recipient": None, "subject": None, "body": None, "confirm": False}
                return "Email cancelled.", random.choice(prompts.CANCEL_SEND_VARIANTS)

    # New command parsing
    try:
        recipient, subject, body = None, None, None

        # Look for "to <recipient>"
        match_to = re.search(r"to ([^ ]+)", text, re.IGNORECASE)
        if match_to:
            recipient = match_to.group(1)

        # Look for "subject <...> body <...>"
        if "subject" in text.lower() and "body" in text.lower():
            subject_part = text.split("subject", 1)[1]
            if "body" in subject_part:
                subject, body = subject_part.split("body", 1)
                subject, body = subject.strip(), body.strip()

        # Look for "about <...>: <...>" pattern
        elif "about" in text.lower() and ":" in text:
            about_part = text.split("about", 1)[1]
            subject, body = about_part.split(":", 1)
            subject, body = subject.strip(), body.strip()

        if recipient and subject and body:
            confirm_prompt = random.choice(prompts.ASK_CONFIRM_VARIANTS).format(
                subject=subject, body=body
            )
            email_pending.update({
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "confirm": True
            })
            return None, confirm_prompt
        elif recipient and not subject:
            email_pending["recipient"] = recipient
            return None, random.choice(prompts.ASK_SUBJECT_VARIANTS)
        else:
            return "I couldn't parse the full email. Please specify recipient, subject, and body.", None

    except Exception as e:
        return f"Error parsing send email command: {e}", None