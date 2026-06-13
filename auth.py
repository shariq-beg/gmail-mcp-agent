# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# This section imports Google OAuth helpers used to load and refresh Gmail credentials.

import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


# -----------------------------------------------------------------------------
# Gmail OAuth Configuration
# -----------------------------------------------------------------------------
# This section defines the Gmail OAuth scopes requested by the application.

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


# -----------------------------------------------------------------------------
# Credential Loading
# -----------------------------------------------------------------------------
# This section loads, refreshes, or creates the token used by Gmail client code.

def get_credentials():
    """Load, refresh, or create OAuth credentials for Gmail API access.
    Args:
        None.
    Returns:
        A Credentials object authorized for the configured Gmail scopes.
    Side effects:
        May open a local OAuth flow and write token.json when credentials change.
    Used by:
        gmail_client.get_service when building the Gmail API service.
    """
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds
