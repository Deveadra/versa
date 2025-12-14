1. **Enable Google Calendar API**
   - Go to [Google Cloud Console](https://console.cloud.google.com/).
   - Create a new project (or select an existing one).
   - Navigate to **APIs & Services → Library**.
   - Search for **Google Calendar API** and click **Enable**.

2. **Create OAuth Credentials**
   - In **APIs & Services → Credentials**, click **Create Credentials → OAuth client ID**.
   - Choose **Desktop App**.
   - Download the resulting `credentials.json` file.
   - Place `credentials.json` in your project root (same folder as `main.py`).

3. **Install Dependencies**

   ```bash
   .\.venv\Scripts\activate    
   pip install -r requirements.txt
   ```

4. **Run Jarvis for the First Time**

   ```bash
   python main.py
   ```

   - On first use, a browser window will open asking you to log in with your Google account.
   - Approve access to your Google Calendar.
   - A `token.pickle` file will be created locally to store your access/refresh token.

5. **Test Commands**
   - "Jarvis, what’s on my calendar?" → retrieves upcoming events.
   - "Jarvis, add event Lunch tomorrow at 1pm" → creates a real event in Google Calendar.

6. **Fallback Mode**
   - If `credentials.json` or internet access is missing, Jarvis will fall back to the mock in-memory calendar.

---
✅ You are now ready to use Google Calendar with Jarvis.


# Gmail Integration Setup

1. **Enable Gmail API**

- Go to [Google Cloud Console](https://console.cloud.google.com/).
- In the same project (or a new one), navigate to **APIs & Services → Library**.
- Search for **Gmail API** and click **Enable**.

2. **OAuth Credentials**

- You do **not** manually edit your `credentials.json`. Scopes are requested at runtime in your code.
- If you already created OAuth credentials for Calendar, you can reuse that same `credentials.json`.
- In your code, when you build the OAuth flow, include Gmail scopes in the `SCOPES` list, for example:

```python
SCOPES = [
'https://www.googleapis.com/auth/calendar',
'https://www.googleapis.com/auth/gmail.modify'
]
```

- On first run, the Google consent screen will show both Calendar and Gmail permissions. Once you approve, your `token.pickle` (or `token_gmail.pickle`) will contain access for both APIs.

3. **Install Dependencies**

- Already included in `requirements.txt`.

4. **Run and Authenticate**

- The first time you call a Gmail function, Jarvis will open a browser window.
- Log in and grant access to Gmail (and Calendar if both scopes are present).
- A token file (e.g., `token_gmail.pickle`) will be created locally to store your credentials.

5. **Test Commands**

- "Jarvis, check my email" → retrieves your latest unread emails.
- "Jarvis, send email to John subject Meeting body See you at 3" → sends an email.

6. **Fallback Mode**

- If Gmail credentials are not present, Jarvis will respond with a mock/fallback message.

---
✅ You are now ready to use Gmail with Jarvis.
