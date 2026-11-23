"""Google sign-in is gone: the Sheets sync and the Drive uploads were retired,
and the app's own UI is the interface. Nothing may bring an OAuth token back
quietly: no import of the sign-in libraries, no helper that uses them, no
package for them."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIGN_IN = re.compile(r"^\s*(?:from|import)\s+(?:gspread|oauth2client|google_auth_oauthlib|googleapiclient|google\.oauth2)\b", re.M)
PACKAGES = ("gspread", "oauth2client", "google-auth-oauthlib", "google-api-python-client", "google-auth-httplib2")


def _py_files():
    for folder in ("src", "scripts"):
        yield from (ROOT / folder).rglob("*.py")


def test_no_code_imports_a_google_sign_in_library():
    hits = [str(p.relative_to(ROOT)) for p in _py_files() if SIGN_IN.search(p.read_text(errors="replace"))]
    assert hits == []


def test_no_google_sign_in_helpers_left():
    gone = ("get_google_credentials", "upload_to_google_sheets", "check_google_token", "oauth_reauth")
    hits = [f"{p.relative_to(ROOT)}: {name}" for p in _py_files()
            for name in gone if name in p.read_text(errors="replace")]
    assert hits == []
    assert not (ROOT / "scripts" / "oauth_reauth.py").exists()
    assert not (ROOT / "src" / "job_agent" / "authenticate_google.py").exists()


def test_no_google_sign_in_packages_in_the_requirements():
    for name in ("requirements.txt", "pyproject.toml"):
        text = (ROOT / name).read_text().lower()
        assert [p for p in PACKAGES if re.search(rf"^{re.escape(p)}\s*[>=~^ ]", text, re.M)] == [], name
