"""Google Sheets OAuth bootstrap for ``tools/steering_to_gsheet.py``.

Thin wrapper around :func:`gspread.oauth` so the CLI tool doesn't have
to know where the credentials live or what scopes we need.  Roger's
one-time Google Cloud setup is documented in ``AGENT_NOTES.md``
("Exporting steering experiments to Google Sheets").

Credential file layout (canonical, both gitignored at the repo root):

    ~/.config/assistant-axis/google_credentials.json   # OAuth client (downloaded from console)
    ~/.config/assistant-axis/google_token.json         # auto-refreshing user token

The first call to :func:`get_client` with a fresh ``google_credentials.json``
pops a browser tab and writes the user token to disk.  Subsequent calls
silently reuse / refresh it.  No service-account dance, no impersonation
-- the OAuth user is whoever Roger logged in as.

Scopes are tight on purpose:

  - ``spreadsheets``  -- needed for read+write of cell values,
                         frozen rows, dimension groups, conditional
                         formats, column widths via batchUpdate.
  - ``drive.file``    -- file-scoped (not "all of Drive"): lets the
                         tool create + read spreadsheets it owns, but
                         not browse the user's Drive.  Sufficient for
                         ``--spreadsheet-name`` Drive-search + create.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path("~/.config/assistant-axis").expanduser()
DEFAULT_CREDS_PATH = DEFAULT_CONFIG_DIR / "google_credentials.json"
DEFAULT_TOKEN_PATH = DEFAULT_CONFIG_DIR / "google_token.json"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]


class MissingCredentialsError(RuntimeError):
    """Raised when no OAuth client secret file is on disk.

    Triggers the user-facing setup-instructions message in the CLI
    rather than the raw gspread/google-auth exception chain.
    """


def get_client(
    creds_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
):
    """Return an authenticated ``gspread.Client``.

    Args:
        creds_path: Override for the OAuth client JSON.  Defaults to
            ``~/.config/assistant-axis/google_credentials.json``.
        token_path: Override for the cached user token.  Defaults to
            ``~/.config/assistant-axis/google_token.json``.  Created on
            first run, refreshed automatically thereafter.

    Raises:
        MissingCredentialsError: when ``creds_path`` doesn't exist on
            disk.  The CLI surfaces this with a pointer to the setup
            instructions in AGENT_NOTES.md.
    """
    # gspread is an optional dependency for the steering pipeline at
    # large -- import lazily so unrelated imports of this module from
    # ``assistant_axis`` don't fail on a stripped-down install.
    import gspread

    creds_path = Path(creds_path) if creds_path else DEFAULT_CREDS_PATH
    token_path = Path(token_path) if token_path else DEFAULT_TOKEN_PATH

    if not creds_path.exists():
        raise MissingCredentialsError(
            f"Google OAuth client secret not found at {creds_path}.\n"
            f"One-time setup:\n"
            f"  1. Visit https://console.cloud.google.com and pick or create a project.\n"
            f"  2. Enable the Google Sheets API and Google Drive API.\n"
            f"  3. Configure OAuth consent screen (External, add yourself as test user).\n"
            f"  4. Credentials -> Create OAuth 2.0 Client ID -> Desktop app.\n"
            f"  5. Download the JSON and save it as {creds_path}.\n"
            f"See AGENT_NOTES.md (\"Exporting steering experiments to Google Sheets\") "
            f"for details."
        )

    # Ensure the parent dir exists so gspread can write the token file.
    token_path.parent.mkdir(parents=True, exist_ok=True)

    return gspread.oauth(
        credentials_filename=str(creds_path),
        authorized_user_filename=str(token_path),
        scopes=SCOPES,
    )
