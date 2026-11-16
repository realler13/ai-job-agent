import logging
import sys
from job_agent.utils import setup_logging, get_google_credentials

# Setup logging
setup_logging(verbose=True)
logger = logging.getLogger("google_auth")

def authenticate():
    print("\n🔐 Starting Google Authentication...")
    print("This will open your browser to login with your Google Account.")
    print("Ensure 'secrets/credentials.json' exists before proceeding.\n")
    
    try:
        creds = get_google_credentials(interactive=True)
        if creds and creds.valid:
            print("\n✅ Authentication Successful! Token saved to 'secrets/token.json'.")
            print("You can now use Google Drive/Sheets features.")
        else:
            print("\n❌ Authentication Failed or Cancelled.")
    except Exception as e:
        logger.error(f"Auth Error: {e}")

if __name__ == "__main__":
    authenticate()
