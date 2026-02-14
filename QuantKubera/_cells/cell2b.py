# ============================================================================
# CELL 2b: Cross-Asset Data — News Sentiment + India VIX
# ============================================================================
#
# Cross-asset features: COMMON across all tickers (one value per day).
# Captures market-wide regime through:
#   1. FinBERT sentiment of GDELT financial news headlines (9 features)
#   2. India VIX fear gauge (3 features)
#
# Architecture:
#   - Headlines fetched from GDELT DOC 2.0 API (free, no API key)
#   - Scored with ProsusAI/finbert on GPU (~5ms/headline)
#   - All data cached to disk for instant re-runs
#   - T+1 causality: headlines from day D -> features for day D+1
#   - India VIX fetched via Kite API (already authenticated)
# ============================================================================

# ---------------------------------------------------------------------------
# FinBERT availability check
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_FINBERT = True
except ImportError:
    HAS_FINBERT = False
    logger.warning("torch/transformers not installed — sentiment features disabled")

# ---------------------------------------------------------------------------
# Cache directories
# ---------------------------------------------------------------------------
_QK_CACHE_DIR = Path.home() / '.quantkubera'
_HEADLINE_CACHE_DIR = _QK_CACHE_DIR / 'headlines'
_SCORE_CACHE_DIR = _QK_CACHE_DIR / 'sentiment_scores'

# ---------------------------------------------------------------------------
# GDELT DOC 2.0 API
# ---------------------------------------------------------------------------
_GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_DELAY = 2.0  # seconds between requests

# Focused queries: India markets + commodities + global macro + crypto
_GDELT_QUERIES = [
    # India markets — covers NIFTY, BANKNIFTY, Sensex, broad market
    '"NIFTY" OR "Sensex" OR "BSE" OR "NSE" OR "Indian stock market" '
    'sourceCountry:IN sourcelang:eng',
    # India FnO stocks
    '"Reliance" OR "TCS" OR "HDFC" OR "Infosys" OR "ICICI" OR "SBI" '
    'sourceCountry:IN sourcelang:eng',
    # Commodities — gold, oil, metals (relevant for MCX tickers)
    '"gold price" OR "crude oil" OR "silver price" OR "copper" OR "natural gas" '
    'sourcelang:eng',
    # Global macro — Fed, rates, trade (affects all markets)
    '"Federal Reserve" OR "interest rate" OR "global markets" OR "trade war" '
    'OR "tariffs" sourcelang:eng',
    # Crypto
    '"bitcoin" OR "ethereum" OR "cryptocurrency" sourcelang:eng',
]


def _gdelt_fetch_articles(query: str, start_dt: datetime,
                           end_dt: datetime) -> list:
    """Fetch articles from GDELT DOC 2.0 API (single request, max 250)."""
    params = {
        'query': query,
        'mode': 'ArtList',
        'format': 'json',
        'maxrecords': 250,
        'startdatetime': start_dt.strftime('%Y%m%d%H%M%S'),
        'enddatetime': end_dt.strftime('%Y%m%d%H%M%S'),
        'sort': 'DateDesc',
    }
    for attempt in range(3):
        try:
            resp = requests.get(_GDELT_ENDPOINT, params=params, timeout=30,
                                headers={'User-Agent': 'QuantKubera/2.1'})
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get('articles', [])
        except requests.exceptions.JSONDecodeError:
            return []
        except Exception as e:
            logger.warning(f"GDELT fetch failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(3)
    return []


def _parse_gdelt_date(seendate: str) -> Optional[datetime]:
    """Parse GDELT seendate like '20260210T083000Z'."""
    try:
        clean = seendate.replace('T', '').replace('Z', '')
        return datetime.strptime(clean, '%Y%m%d%H%M%S')
    except (ValueError, AttributeError):
        return None


def fetch_gdelt_headlines(start_date: str, end_date: str,
                          chunk_days: int = 14) -> Dict[str, list]:
    """Fetch financial news headlines from GDELT with disk caching.

    Args:
        start_date, end_date: 'YYYY-MM-DD'
        chunk_days: calendar days per API request

    Returns: {date_str: [{'title': str, 'source': str}]}
    """
    _HEADLINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check disk cache
    cache_file = _HEADLINE_CACHE_DIR / f"gdelt_{start_date}_{end_date}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            n_total = sum(len(v) for v in cached.values())
            print(f"  GDELT cache hit: {n_total} headlines across "
                  f"{len(cached)} days")
            return cached
        except (json.JSONDecodeError, KeyError):
            pass

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(
        hour=23, minute=59, second=59)

    seen_titles = set()
    daily_headlines: Dict[str, list] = {}
    total = 0

    n_queries = len(_GDELT_QUERIES)
    n_chunks_per_query = max(1, (end_dt - start_dt).days // chunk_days + 1)
    total_requests = n_queries * n_chunks_per_query

    pbar = tqdm(total=total_requests, desc="GDELT headlines",
                unit="req", leave=False)

    for q_idx, query in enumerate(_GDELT_QUERIES):
        chunk_start = start_dt
        while chunk_start < end_dt:
            chunk_end = min(
                chunk_start + timedelta(days=chunk_days), end_dt)

            articles = _gdelt_fetch_articles(query, chunk_start, chunk_end)

            for art in articles:
                title = art.get('title', '').strip()
                if not title or len(title) < 15:
                    continue

                norm = title.lower().strip()
                if norm in seen_titles:
                    continue
                seen_titles.add(norm)

                dt = _parse_gdelt_date(art.get('seendate', ''))
                if dt is None:
                    continue

                date_str = dt.strftime('%Y-%m-%d')
                if date_str not in daily_headlines:
                    daily_headlines[date_str] = []

                daily_headlines[date_str].append({
                    'title': title,
                    'source': art.get('domain', 'unknown'),
                })
                total += 1

            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(_GDELT_DELAY)
            pbar.update(1)
            pbar.set_postfix_str(f"{total} headlines")

    pbar.close()

    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(daily_headlines, f)

    print(f"  GDELT: {total} unique headlines across {len(daily_headlines)} days")
    return daily_headlines


# ---------------------------------------------------------------------------
# FinBERT Sentiment Scorer with disk caching
# ---------------------------------------------------------------------------
_finbert_model = None
_finbert_tokenizer = None
_finbert_device = None


def _load_finbert():
    """Load ProsusAI/finbert model (singleton, GPU if available)."""
    global _finbert_model, _finbert_tokenizer, _finbert_device
    if _finbert_model is not None:
        return

    if not HAS_FINBERT:
        raise RuntimeError("torch/transformers not installed")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Loading ProsusAI/finbert on {device}...")

    _finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    _finbert_model = AutoModelForSequenceClassification.from_pretrained(
        'ProsusAI/finbert')
    _finbert_model.to(device)
    _finbert_model.eval()
    _finbert_device = device
    print(f"  FinBERT loaded on {device}")


def _finbert_score_batch(texts: list, batch_size: int = 64) -> list:
    """Score texts with FinBERT. Returns [(score, confidence, label), ...]

    score: float in [-1, +1] (positive - negative probability)
    confidence: float in [0, 1] (max class probability)
    label: str "positive" | "negative" | "neutral"
    """
    _load_finbert()
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = _finbert_tokenizer(
            batch, padding=True, truncation=True,
            max_length=128, return_tensors='pt'
        ).to(_finbert_device)

        with torch.no_grad():
            outputs = _finbert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        for j in range(len(batch)):
            p = probs[j]
            pos, neg = p[0].item(), p[1].item()
            score = pos - neg
            confidence = p.max().item()
            label = label_map[p.argmax().item()]
            results.append((score, confidence, label))

    return results


def _title_hash(title: str) -> str:
    """Deterministic hash for cache key."""
    return hashlib.sha256(title.strip().lower().encode()).hexdigest()[:16]


def score_headlines_with_cache(
    daily_headlines: Dict[str, list],
) -> Dict[str, list]:
    """Score all headlines with FinBERT, using disk cache.

    Args:
        daily_headlines: {date_str: [{'title': str, 'source': str}]}

    Returns: {date_str: [(score, confidence, label), ...]}
    """
    if not HAS_FINBERT:
        return {}

    _SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _SCORE_CACHE_DIR / 'finbert_scores.json'

    # Load existing cache {title_hash: [score, confidence, label]}
    score_cache: Dict[str, list] = {}
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                score_cache = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    # Collect uncached titles
    uncached = []  # (date_str, idx, title, hash)
    for date_str, headlines in daily_headlines.items():
        for idx, h in enumerate(headlines):
            th = _title_hash(h['title'])
            if th not in score_cache:
                uncached.append((date_str, idx, h['title'], th))

    if uncached:
        print(f"  Scoring {len(uncached)} uncached headlines with FinBERT...")
        titles = [u[2] for u in uncached]
        scores = _finbert_score_batch(titles)

        for (date_str, idx, title, th), (score, conf, label) in zip(
                uncached, scores):
            score_cache[th] = [score, conf, label]

        # Persist cache
        with open(cache_file, 'w') as f:
            json.dump(score_cache, f)
        print(f"  Scored {len(uncached)} headlines, cache now "
              f"{len(score_cache)} entries")
    else:
        print(f"  All {sum(len(v) for v in daily_headlines.values())} "
              f"headlines already cached")

    # Build result
    result: Dict[str, list] = {}
    for date_str, headlines in daily_headlines.items():
        day_scores = []
        for h in headlines:
            th = _title_hash(h['title'])
            if th in score_cache:
                day_scores.append(tuple(score_cache[th]))
            else:
                day_scores.append((0.0, 0.5, 'neutral'))
        result[date_str] = day_scores

    return result


# ---------------------------------------------------------------------------
# Sentiment Feature Builder (9 features, T+1 lag)
# ---------------------------------------------------------------------------
SENTIMENT_COLUMNS = [
    'ns_sent_mean', 'ns_sent_std', 'ns_pos_ratio', 'ns_neg_ratio',
    'ns_confidence_mean', 'ns_news_count',
    'ns_sent_5d_ma', 'ns_news_count_5d', 'ns_sent_momentum',
]


def build_sentiment_features(
    daily_scores: Dict[str, list],
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build 9 daily sentiment features with T+1 causal lag.

    Headlines from day D -> features for day D+1 (zero look-ahead).

    Args:
        daily_scores: {date_str: [(score, confidence, label), ...]}
        date_index: DatetimeIndex to align features to

    Returns: DataFrame with SENTIMENT_COLUMNS, indexed by date
    """
    if not daily_scores:
        return pd.DataFrame(columns=SENTIMENT_COLUMNS,
                            index=date_index, dtype=float)

    rows = []
    for date_str, scores in sorted(daily_scores.items()):
        if not scores:
            continue
        scores_arr = np.array([s[0] for s in scores])
        confs_arr = np.array([s[1] for s in scores])
        labels = [s[2] for s in scores]
        n = len(scores_arr)
        n_pos = sum(1 for l in labels if l == 'positive')
        n_neg = sum(1 for l in labels if l == 'negative')

        rows.append({
            'date': pd.Timestamp(date_str),
            'ns_sent_mean': float(np.mean(scores_arr)),
            'ns_sent_std': float(np.std(scores_arr, ddof=1)) if n > 1 else 0.0,
            'ns_pos_ratio': n_pos / n if n > 0 else 0.0,
            'ns_neg_ratio': n_neg / n if n > 0 else 0.0,
            'ns_confidence_mean': float(np.mean(confs_arr)),
            'ns_news_count': float(n),
        })

    if not rows:
        return pd.DataFrame(columns=SENTIMENT_COLUMNS,
                            index=date_index, dtype=float)

    df = pd.DataFrame(rows).set_index('date').sort_index()

    # Rolling features (causal — use only past data)
    df['ns_sent_5d_ma'] = df['ns_sent_mean'].rolling(5, min_periods=1).mean()
    df['ns_news_count_5d'] = df['ns_news_count'].rolling(5, min_periods=1).sum()
    df['ns_sent_momentum'] = df['ns_sent_mean'] - df['ns_sent_5d_ma']

    # T+1 lag: headlines from day D become features for day D+1
    df = df.shift(1)

    # Align to target date index, forward-fill short gaps (weekends)
    df = df.reindex(date_index)
    df = df.ffill(limit=3)

    return df[SENTIMENT_COLUMNS]


# ---------------------------------------------------------------------------
# India VIX Fetcher (via Kite API)
# ---------------------------------------------------------------------------
def fetch_india_vix_kite(kite: KiteConnect, lookback_days: int = 2500
                         ) -> pd.DataFrame:
    """Fetch India VIX daily OHLCV via Kite API.

    Uses the authenticated KiteConnect instance to fetch INDIA VIX from NSE.

    Args:
        kite: authenticated KiteConnect instance
        lookback_days: calendar days of history

    Returns: DataFrame with [vix_close, vix_open, vix_high, vix_low],
             index=date (tz-naive DatetimeIndex)
    """
    empty = pd.DataFrame(
        columns=['vix_open', 'vix_high', 'vix_low', 'vix_close'])

    # Resolve INDIA VIX instrument token
    try:
        instruments = pd.DataFrame(kite.instruments('NSE'))
        vix_rows = instruments[
            instruments['tradingsymbol'] == 'INDIA VIX']
        if vix_rows.empty:
            # Try partial match
            vix_rows = instruments[
                instruments['tradingsymbol'].str.contains(
                    'VIX', case=False, na=False)]
        if vix_rows.empty:
            logger.warning("INDIA VIX instrument not found on NSE")
            return empty
        token = int(vix_rows.iloc[0]['instrument_token'])
        symbol = vix_rows.iloc[0]['tradingsymbol']
        print(f"  Resolved VIX: {symbol} (token={token})")
    except Exception as e:
        logger.warning(f"VIX instrument resolution failed: {e}")
        return empty

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=lookback_days)

    all_records = []
    chunk_start = start_date
    max_chunk = 1900

    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=max_chunk), end_date)
        try:
            records = kite.historical_data(
                instrument_token=token,
                from_date=chunk_start,
                to_date=chunk_end,
                interval='day',
                continuous=False,  # VIX is an index, not futures
            )
            all_records.extend(records)
        except Exception as e:
            logger.warning(f"VIX chunk {chunk_start}-{chunk_end} failed: {e}")
        chunk_start = chunk_end + timedelta(days=1)

    if not all_records:
        return empty

    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.set_index('date').sort_index()
    df.columns = [c.lower() for c in df.columns]

    # Rename to vix_* prefix
    rename_map = {'close': 'vix_close', 'open': 'vix_open',
                  'high': 'vix_high', 'low': 'vix_low'}
    df = df.rename(columns=rename_map)
    keep = [c for c in ['vix_open', 'vix_high', 'vix_low', 'vix_close']
            if c in df.columns]
    df = df[keep]
    df = df[~df.index.duplicated(keep='last')]

    print(f"  India VIX: {len(df)} days "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ---------------------------------------------------------------------------
# VIX Feature Builder (3 features)
# ---------------------------------------------------------------------------
VIX_COLUMNS = ['vix_level', 'vix_change_5d', 'vix_mean_reversion']


def build_vix_features(vix_df: pd.DataFrame,
                       date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build 3 VIX features.

    Args:
        vix_df: DataFrame with vix_close column
        date_index: DatetimeIndex to align to

    Returns: DataFrame with VIX_COLUMNS
    """
    if vix_df.empty or 'vix_close' not in vix_df.columns:
        return pd.DataFrame(columns=VIX_COLUMNS,
                            index=date_index, dtype=float)

    vix = vix_df['vix_close'].copy()
    result = pd.DataFrame(index=vix.index)

    # VIX z-score: (VIX - 60d mean) / 60d std
    vix_mean = vix.rolling(60, min_periods=20).mean()
    vix_std = vix.rolling(60, min_periods=20).std(ddof=1)
    vix_std = vix_std.replace(0, np.nan)
    result['vix_level'] = (vix - vix_mean) / vix_std

    # 5-day VIX change (percent)
    result['vix_change_5d'] = vix.pct_change(5)

    # VIX mean reversion: (VIX - 20d MA) / 20d std
    vix_ma20 = vix.rolling(20, min_periods=10).mean()
    vix_std20 = vix.rolling(20, min_periods=10).std(ddof=1)
    vix_std20 = vix_std20.replace(0, np.nan)
    result['vix_mean_reversion'] = (vix - vix_ma20) / vix_std20

    # Align to target date index, forward-fill short gaps
    result = result.reindex(date_index)
    result = result.ffill(limit=3)

    return result[VIX_COLUMNS]


# ---------------------------------------------------------------------------
# Master Cross-Asset Function
# ---------------------------------------------------------------------------
CROSS_ASSET_COLUMNS: List[str] = []  # populated at runtime


def fetch_cross_asset_features(
    start_date: str,
    end_date: str,
    date_index: pd.DatetimeIndex,
    kite: Optional[KiteConnect] = None,
) -> pd.DataFrame:
    """Fetch all cross-asset data and build features.

    Populates the global CROSS_ASSET_COLUMNS with available feature names.

    Args:
        start_date, end_date: 'YYYY-MM-DD'
        date_index: DatetimeIndex to align all features to
        kite: authenticated KiteConnect (for VIX)

    Returns: DataFrame with available cross-asset features
    """
    global CROSS_ASSET_COLUMNS

    all_features = pd.DataFrame(index=date_index)
    available_cols: List[str] = []

    # --- News Sentiment (GDELT + FinBERT) ---
    print("\n  [1/2] News Sentiment (GDELT + FinBERT)")
    try:
        daily_headlines = fetch_gdelt_headlines(start_date, end_date)

        if daily_headlines and HAS_FINBERT:
            daily_scores = score_headlines_with_cache(daily_headlines)
            sent_df = build_sentiment_features(daily_scores, date_index)

            for col in SENTIMENT_COLUMNS:
                if col in sent_df.columns:
                    all_features[col] = sent_df[col]
                    available_cols.append(col)

            n_days = sent_df.notna().any(axis=1).sum()
            print(f"    -> {len(available_cols)} features, "
                  f"{n_days} days with data")
        elif not HAS_FINBERT:
            print("    SKIP: FinBERT not available "
                  "(pip install torch transformers)")
        else:
            print("    SKIP: No headlines fetched from GDELT")
    except Exception as e:
        logger.warning(f"Sentiment features failed: {e}")
        print(f"    SKIP: Failed ({e})")

    # --- India VIX ---
    print("  [2/2] India VIX")
    try:
        if kite is not None:
            vix_df = fetch_india_vix_kite(kite)
            if not vix_df.empty:
                vix_feats = build_vix_features(vix_df, date_index)
                for col in VIX_COLUMNS:
                    if col in vix_feats.columns:
                        all_features[col] = vix_feats[col]
                        available_cols.append(col)
                n_days = vix_feats.notna().any(axis=1).sum()
                print(f"    -> {len(VIX_COLUMNS)} features, "
                      f"{n_days} days with data")
            else:
                print("    SKIP: No VIX data from Kite")
        else:
            print("    SKIP: No Kite instance provided")
    except Exception as e:
        logger.warning(f"VIX features failed: {e}")
        print(f"    SKIP: Failed ({e})")

    CROSS_ASSET_COLUMNS = available_cols

    print(f"\n  Cross-Asset Summary: {len(available_cols)} features available")
    if available_cols:
        print(f"    Columns: {available_cols}")

    if available_cols:
        return all_features[available_cols]
    return pd.DataFrame(index=date_index)
