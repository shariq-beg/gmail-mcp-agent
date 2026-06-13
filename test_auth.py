# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# This section imports the credential helper used by the authentication smoke test.

from auth import get_credentials


# -----------------------------------------------------------------------------
# Authentication Smoke Test
# -----------------------------------------------------------------------------
# This section checks that Gmail credentials can be loaded and reports token validity.

creds = get_credentials()
print("Authentication successful")
print("Token valid:", creds.valid)
