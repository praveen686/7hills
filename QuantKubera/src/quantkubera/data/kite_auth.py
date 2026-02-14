import os
import json
import logging
import pyotp
import requests
from kiteconnect import KiteConnect
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class KiteAuth:
    def __init__(self, env_path=None):
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
            
        self.api_key = os.getenv("ZERODHA_API_KEY")
        self.api_secret = os.getenv("ZERODHA_API_SECRET")
        self.totp_secret = os.getenv("ZERODHA_TOTP_SECRET")
        self.user_id = os.getenv("ZERODHA_USER_ID")
        self.password = os.getenv("ZERODHA_PASSWORD")
        self.token_file = os.path.expanduser("~/.zerodha_access_token")

        if not all([self.api_key, self.api_secret, self.totp_secret, self.user_id, self.password]):
            logger.warning("Missing one or more Zerodha environment variables.")

    def get_session(self):
        """Returns an authenticated KiteConnect session."""
        # Try loading existing token
        if os.path.exists(self.token_file):
            with open(self.token_file, 'r') as f:
                access_token = f.read().strip()
            
            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(access_token)
            try:
                # verify token is valid
                kite.profile()
                return kite
            except Exception:
                logger.info("Saved access token expired or invalid.")

        # Perform full login if cached token fails
        access_token = self.authenticate()
        kite = KiteConnect(api_key=self.api_key)
        kite.set_access_token(access_token)
        return kite

    def authenticate(self):
        """Performs full authentication flow with TOTP."""
        logger.info("Starting new authentication flow...")
        totp = pyotp.TOTP(self.totp_secret).now()
        
        with requests.Session() as session:
            # Step 1: Login with credentials
            resp = session.post("https://kite.zerodha.com/api/login", data={
                "user_id": self.user_id,
                "password": self.password
            })
            resp.raise_for_status()
            login_data = resp.json()

            # Step 2: 2FA with TOTP
            resp = session.post("https://kite.zerodha.com/api/twofa", data={
                "user_id": self.user_id,
                "request_id": login_data["data"]["request_id"],
                "twofa_value": totp
            })
            resp.raise_for_status()

            # Step 3: Get request_token
            kite_login_url = f"https://kite.trade/connect/login?api_key={self.api_key}"
            resp = session.get(kite_login_url)
            request_token = resp.url.split("request_token=")[1].split("&")[0]

            # Step 4: Finalize session
            kite = KiteConnect(api_key=self.api_key)
            session_data = kite.generate_session(request_token, api_secret=self.api_secret)
            access_token = session_data["access_token"]

            # Save token
            with open(self.token_file, 'w') as f:
                f.write(access_token)
            
            return access_token
