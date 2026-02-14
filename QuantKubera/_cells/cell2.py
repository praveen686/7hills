# ============================================================================
# CELL 2: Data Engine — KiteAuth + KiteFetcher
# ============================================================================

load_dotenv()


class KiteAuth:
    """TOTP-based Zerodha authentication with token caching."""

    TOKEN_PATH = Path.home() / '.zerodha_access_token'
    LOGIN_URL = 'https://kite.zerodha.com/api/login'
    TWOFA_URL = 'https://kite.zerodha.com/api/twofa'
    CONNECT_URL = 'https://kite.zerodha.com/connect/login'

    def __init__(self):
        self.api_key = os.getenv('ZERODHA_API_KEY', '')
        self.api_secret = os.getenv('ZERODHA_API_SECRET', '')
        self.totp_secret = os.getenv('ZERODHA_TOTP_SECRET', '')
        self.user_id = os.getenv('ZERODHA_USER_ID', '')
        self.password = os.getenv('ZERODHA_PASSWORD', '')

    def _load_cached_token(self) -> Optional[str]:
        """Load cached access token if it exists and was created today."""
        if not self.TOKEN_PATH.exists():
            return None
        stat = self.TOKEN_PATH.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).date()
        if modified != datetime.now().date():
            return None
        try:
            token_data = json.loads(self.TOKEN_PATH.read_text())
            return token_data.get('access_token')
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_token(self, access_token: str):
        """Cache the access token to disk."""
        self.TOKEN_PATH.write_text(json.dumps({
            'access_token': access_token,
            'timestamp': datetime.now().isoformat(),
            'user_id': self.user_id,
        }))

    def _auto_login(self) -> str:
        """Perform full auto-login flow and return access token."""
        session = requests.Session()

        # Step 1: POST /api/login
        logger.info("KiteAuth: Step 1 — login")
        resp = session.post(self.LOGIN_URL, data={
            'user_id': self.user_id,
            'password': self.password,
        })
        resp.raise_for_status()
        login_data = resp.json()
        if login_data.get('status') != 'success':
            raise RuntimeError(f"Login failed: {login_data}")
        request_id = login_data['data']['request_id']

        # Step 2: POST /api/twofa with TOTP
        logger.info("KiteAuth: Step 2 — TOTP 2FA")
        totp = pyotp.TOTP(self.totp_secret)
        twofa_value = totp.now()
        resp = session.post(self.TWOFA_URL, data={
            'user_id': self.user_id,
            'request_id': request_id,
            'twofa_value': twofa_value,
            'twofa_type': 'totp',
        })
        resp.raise_for_status()
        twofa_data = resp.json()
        if twofa_data.get('status') != 'success':
            raise RuntimeError(f"2FA failed: {twofa_data}")

        # Step 3: GET /connect/login to extract request_token
        # Follow the full redirect chain — request_token appears in the
        # final redirect to the registered callback URL.
        logger.info("KiteAuth: Step 3 — extract request_token")
        from urllib.parse import urlparse, parse_qs

        resp = session.get(self.CONNECT_URL, params={
            'api_key': self.api_key,
            'v': '3',
        }, allow_redirects=True)

        # Search for request_token in the final URL and all redirect history
        candidate_urls = [resp.url] + [r.headers.get('Location', '') for r in resp.history]
        request_token = None
        for url in candidate_urls:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'request_token' in params:
                request_token = params['request_token'][0]
                break

        if request_token is None:
            raise RuntimeError(
                f"Could not extract request_token. "
                f"Final URL: {resp.url}, "
                f"History: {[r.url for r in resp.history]}"
            )

        # Step 4: Generate session
        logger.info("KiteAuth: Step 4 — generate session")
        kite = KiteConnect(api_key=self.api_key)
        data = kite.generate_session(request_token, api_secret=self.api_secret)
        access_token = data['access_token']

        self._save_token(access_token)
        logger.info("KiteAuth: login complete, token cached")
        return access_token

    def get_session(self) -> KiteConnect:
        """Return an authenticated KiteConnect instance."""
        # Try cached token first
        cached = self._load_cached_token()
        if cached:
            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(cached)
            try:
                kite.profile()
                logger.info("KiteAuth: using cached token")
                return kite
            except Exception:
                logger.info("KiteAuth: cached token expired, re-logging in")

        # Full login
        access_token = self._auto_login()
        kite = KiteConnect(api_key=self.api_key)
        kite.set_access_token(access_token)
        return kite


class KiteFetcher:
    """Fetches daily OHLCV data from Zerodha Kite."""

    MAX_CHUNK_DAYS = 1900  # API limit is 2000, use 1900 for safety

    def __init__(self, kite: KiteConnect):
        self.kite = kite
        self._instruments_cache: Optional[pd.DataFrame] = None

    def _get_instruments(self, exchange: str) -> pd.DataFrame:
        """Fetch and cache instruments list for the given exchange."""
        if self._instruments_cache is None:
            all_instruments = self.kite.instruments()
            self._instruments_cache = pd.DataFrame(all_instruments)
        return self._instruments_cache[
            self._instruments_cache['exchange'] == exchange
        ].copy()

    def _resolve_instrument(self, symbol: str, exchange: str) -> Tuple[str, str]:
        """
        Resolve symbol to instrument_token.
        Try exact match first, then fuzzy match for derivatives (FUT, nearest expiry).
        Returns (instrument_token, resolved_tradingsymbol).
        """
        instruments = self._get_instruments(exchange)

        # Exact match on tradingsymbol
        exact = instruments[instruments['tradingsymbol'] == symbol]
        if len(exact) > 0:
            row = exact.iloc[0]
            return str(row['instrument_token']), row['tradingsymbol']

        # Try NFO/MCX-FUT: look for symbol + FUT with nearest expiry
        # For index derivatives, check NFO exchange
        fut_exchange = exchange
        if exchange == 'NSE':
            fut_exchange = 'NFO'
            instruments = self._get_instruments(fut_exchange)
        elif exchange == 'MCX':
            # MCX futures are on MCX itself
            pass

        # Fuzzy match: tradingsymbol starts with symbol, segment contains FUT
        fuzzy = instruments[
            instruments['tradingsymbol'].str.startswith(symbol) &
            (instruments['instrument_type'] == 'FUT')
        ].copy()

        if len(fuzzy) == 0:
            # Try with exchange-specific naming
            fuzzy = instruments[
                instruments['tradingsymbol'].str.contains(symbol, case=False, na=False) &
                (instruments['instrument_type'] == 'FUT')
            ].copy()

        if len(fuzzy) == 0:
            raise ValueError(
                f"Could not resolve instrument: {symbol} on {exchange}/{fut_exchange}. "
                f"Check symbol name and exchange."
            )

        # Pick the nearest expiry
        if 'expiry' in fuzzy.columns:
            fuzzy['expiry'] = pd.to_datetime(fuzzy['expiry'], errors='coerce')
            fuzzy = fuzzy.dropna(subset=['expiry'])
            fuzzy = fuzzy.sort_values('expiry')
            # Pick the nearest future expiry (>= today)
            today = pd.Timestamp.now().normalize()
            future_expiries = fuzzy[fuzzy['expiry'] >= today]
            if len(future_expiries) > 0:
                row = future_expiries.iloc[0]
            else:
                row = fuzzy.iloc[-1]  # Fallback to latest
        else:
            row = fuzzy.iloc[0]

        resolved = row['tradingsymbol']
        token = str(row['instrument_token'])
        print(f"  Resolved {symbol} -> {resolved} (expiry: {row.get('expiry', 'N/A')})")
        return token, resolved

    def fetch_daily(self, symbol: str, exchange: str, days: int = 2500) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for the given symbol.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'NIFTY', 'BANKNIFTY', 'SILVER')
        exchange : str
            Exchange (e.g., 'NSE', 'MCX')
        days : int
            Number of calendar days to look back

        Returns
        -------
        pd.DataFrame
            DataFrame with index=date (tz-naive), columns: open, high, low, close, volume
        """
        instrument_token, resolved_name = self._resolve_instrument(symbol, exchange)

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        print(f"  Fetching {resolved_name}: {start_date} to {end_date} ({days} calendar days)")

        all_records = []
        chunk_start = start_date

        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=self.MAX_CHUNK_DAYS), end_date)

            try:
                records = self.kite.historical_data(
                    instrument_token=int(instrument_token),
                    from_date=chunk_start,
                    to_date=chunk_end,
                    interval='day',
                    continuous=True,
                )
                all_records.extend(records)
            except Exception as e:
                logger.warning(f"  Chunk {chunk_start}-{chunk_end} failed: {e}")

            chunk_start = chunk_end + timedelta(days=1)

        if not all_records:
            raise ValueError(f"No data returned for {symbol} ({resolved_name})")

        df = pd.DataFrame(all_records)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.set_index('date').sort_index()

        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV
        keep_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in keep_cols:
            if col not in df.columns:
                df[col] = np.nan

        df = df[keep_cols]
        df = df[~df.index.duplicated(keep='last')]

        print(f"  {resolved_name}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
        return df