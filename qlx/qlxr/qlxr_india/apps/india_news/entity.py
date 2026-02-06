"""Company name → NSE symbol mapping for India F&O stocks.

Maps common names, abbreviations, and informal references to their
canonical NSE ticker symbols. Used to extract stock mentions from
business news headlines.

Only maps to stocks in the F&O universe (apps.india_scanner.universe).
"""

from __future__ import annotations

import re

from apps.india_scanner.universe import FNO_UNIVERSE

# ---------------------------------------------------------------------------
# Company aliases → NSE symbol
# Longest-match-first extraction: "Tata Consultancy" matches before "Tata"
# ---------------------------------------------------------------------------

COMPANY_ALIASES: dict[str, str] = {
    # --- Reliance ---
    "Reliance Industries": "RELIANCE",
    "Reliance Ind": "RELIANCE",
    "Reliance": "RELIANCE",
    "RIL": "RELIANCE",
    # --- TCS ---
    "Tata Consultancy Services": "TCS",
    "Tata Consultancy": "TCS",
    "TCS": "TCS",
    # --- HDFC Bank ---
    "HDFC Bank": "HDFCBANK",
    "HDFCBANK": "HDFCBANK",
    # --- Infosys ---
    "Infosys": "INFY",
    "INFY": "INFY",
    # --- ICICI Bank ---
    "ICICI Bank": "ICICIBANK",
    "ICICIBANK": "ICICIBANK",
    "ICICI": "ICICIBANK",
    # --- HUL ---
    "Hindustan Unilever": "HINDUNILVR",
    "HINDUNILVR": "HINDUNILVR",
    "HUL": "HINDUNILVR",
    # --- SBI ---
    "State Bank of India": "SBIN",
    "State Bank": "SBIN",
    "SBI": "SBIN",
    "SBIN": "SBIN",
    # --- Bharti Airtel ---
    "Bharti Airtel": "BHARTIARTL",
    "BHARTIARTL": "BHARTIARTL",
    "Airtel": "BHARTIARTL",
    # --- ITC ---
    "ITC": "ITC",
    # --- Kotak ---
    "Kotak Mahindra Bank": "KOTAKBANK",
    "Kotak Mahindra": "KOTAKBANK",
    "Kotak Bank": "KOTAKBANK",
    "KOTAKBANK": "KOTAKBANK",
    "Kotak": "KOTAKBANK",
    # --- L&T ---
    "Larsen & Toubro": "LT",
    "Larsen and Toubro": "LT",
    "L&T": "LT",
    "LT": "LT",
    # --- Axis Bank ---
    "Axis Bank": "AXISBANK",
    "AXISBANK": "AXISBANK",
    # --- Bajaj Finance ---
    "Bajaj Finance": "BAJFINANCE",
    "BAJFINANCE": "BAJFINANCE",
    # --- Maruti ---
    "Maruti Suzuki": "MARUTI",
    "Maruti": "MARUTI",
    "MARUTI": "MARUTI",
    # --- HCL Tech ---
    "HCL Technologies": "HCLTECH",
    "HCL Tech": "HCLTECH",
    "HCLTECH": "HCLTECH",
    # --- Sun Pharma ---
    "Sun Pharmaceutical": "SUNPHARMA",
    "Sun Pharma": "SUNPHARMA",
    "SUNPHARMA": "SUNPHARMA",
    # --- Tata Motors ---
    "Tata Motors": "TATAMOTORS",
    "TATAMOTORS": "TATAMOTORS",
    # --- Titan ---
    "Titan Company": "TITAN",
    "Titan": "TITAN",
    "TITAN": "TITAN",
    # --- Asian Paints ---
    "Asian Paints": "ASIANPAINT",
    "ASIANPAINT": "ASIANPAINT",
    # --- UltraTech ---
    "UltraTech Cement": "ULTRACEMCO",
    "UltraTech": "ULTRACEMCO",
    "ULTRACEMCO": "ULTRACEMCO",
    # --- Wipro ---
    "Wipro": "WIPRO",
    "WIPRO": "WIPRO",
    # --- ONGC ---
    "Oil and Natural Gas": "ONGC",
    "ONGC": "ONGC",
    # --- NTPC ---
    "NTPC": "NTPC",
    # --- Power Grid ---
    "Power Grid Corporation": "POWERGRID",
    "Power Grid Corp": "POWERGRID",
    "Power Grid": "POWERGRID",
    "POWERGRID": "POWERGRID",
    # --- Tata Steel ---
    "Tata Steel": "TATASTEEL",
    "TATASTEEL": "TATASTEEL",
    # --- Adani Enterprises ---
    "Adani Enterprises": "ADANIENT",
    "ADANIENT": "ADANIENT",
    # --- Adani Ports ---
    "Adani Ports": "ADANIPORTS",
    "ADANIPORTS": "ADANIPORTS",
    # --- Coal India ---
    "Coal India": "COALINDIA",
    "COALINDIA": "COALINDIA",
    # --- Bajaj Finserv ---
    "Bajaj Finserv": "BAJAJFINSV",
    "BAJAJFINSV": "BAJAJFINSV",
    # --- Tech Mahindra ---
    "Tech Mahindra": "TECHM",
    "TECHM": "TECHM",
    # --- Nestle India ---
    "Nestle India": "NESTLEIND",
    "Nestle": "NESTLEIND",
    "NESTLEIND": "NESTLEIND",
    # --- JSW Steel ---
    "JSW Steel": "JSWSTEEL",
    "JSWSTEEL": "JSWSTEEL",
    "JSW": "JSWSTEEL",
    # --- Grasim ---
    "Grasim Industries": "GRASIM",
    "Grasim": "GRASIM",
    "GRASIM": "GRASIM",
    # --- Divi's Labs ---
    "Divi's Laboratories": "DIVISLAB",
    "Divis Lab": "DIVISLAB",
    "DIVISLAB": "DIVISLAB",
    # --- Cipla ---
    "Cipla": "CIPLA",
    "CIPLA": "CIPLA",
    # --- BPCL ---
    "Bharat Petroleum": "BPCL",
    "BPCL": "BPCL",
    # --- Dr Reddy's ---
    "Dr Reddy's Laboratories": "DRREDDY",
    "Dr Reddy's": "DRREDDY",
    "Dr Reddys": "DRREDDY",
    "DRREDDY": "DRREDDY",
    # --- Eicher Motors ---
    "Eicher Motors": "EICHERMOT",
    "EICHERMOT": "EICHERMOT",
    "Royal Enfield": "EICHERMOT",
    # --- Britannia ---
    "Britannia Industries": "BRITANNIA",
    "Britannia": "BRITANNIA",
    "BRITANNIA": "BRITANNIA",
    # --- Apollo Hospitals ---
    "Apollo Hospitals": "APOLLOHOSP",
    "Apollo Hospital": "APOLLOHOSP",
    "APOLLOHOSP": "APOLLOHOSP",
    # --- Hero MotoCorp ---
    "Hero MotoCorp": "HEROMOTOCO",
    "Hero Moto": "HEROMOTOCO",
    "HEROMOTOCO": "HEROMOTOCO",
    # --- IndusInd Bank ---
    "IndusInd Bank": "INDUSINDBK",
    "INDUSINDBK": "INDUSINDBK",
    "IndusInd": "INDUSINDBK",
    # --- Hindalco ---
    "Hindalco Industries": "HINDALCO",
    "Hindalco": "HINDALCO",
    "HINDALCO": "HINDALCO",
    # --- SBI Life ---
    "SBI Life Insurance": "SBILIFE",
    "SBI Life": "SBILIFE",
    "SBILIFE": "SBILIFE",
    # --- HDFC Life ---
    "HDFC Life Insurance": "HDFCLIFE",
    "HDFC Life": "HDFCLIFE",
    "HDFCLIFE": "HDFCLIFE",
    # --- Tata Consumer ---
    "Tata Consumer Products": "TATACONSUM",
    "Tata Consumer": "TATACONSUM",
    "TATACONSUM": "TATACONSUM",
    # --- Dabur ---
    "Dabur India": "DABUR",
    "Dabur": "DABUR",
    "DABUR": "DABUR",
    # --- Pidilite ---
    "Pidilite Industries": "PIDILITIND",
    "Pidilite": "PIDILITIND",
    "PIDILITIND": "PIDILITIND",
    # --- Godrej Consumer ---
    "Godrej Consumer Products": "GODREJCP",
    "Godrej Consumer": "GODREJCP",
    "GODREJCP": "GODREJCP",
    # --- Havells ---
    "Havells India": "HAVELLS",
    "Havells": "HAVELLS",
    "HAVELLS": "HAVELLS",
    # --- Siemens ---
    "Siemens India": "SIEMENS",
    "Siemens": "SIEMENS",
    "SIEMENS": "SIEMENS",
    # --- Ambuja Cements ---
    "Ambuja Cements": "AMBUJACEM",
    "Ambuja Cement": "AMBUJACEM",
    "Ambuja": "AMBUJACEM",
    "AMBUJACEM": "AMBUJACEM",
    # --- ACC ---
    "ACC Limited": "ACC",
    "ACC": "ACC",
    # --- DLF ---
    "DLF": "DLF",
    # --- Bank of Baroda ---
    "Bank of Baroda": "BANKBARODA",
    "BANKBARODA": "BANKBARODA",
    "BoB": "BANKBARODA",
    # --- PNB ---
    "Punjab National Bank": "PNB",
    "PNB": "PNB",
    # --- Canara Bank ---
    "Canara Bank": "CANBK",
    "CANBK": "CANBK",
    # --- IDFC First Bank ---
    "IDFC First Bank": "IDFCFIRSTB",
    "IDFC First": "IDFCFIRSTB",
    "IDFCFIRSTB": "IDFCFIRSTB",
    # --- Federal Bank ---
    "Federal Bank": "FEDERALBNK",
    "FEDERALBNK": "FEDERALBNK",
    # --- M&M ---
    "Mahindra & Mahindra": "M&M",
    "Mahindra and Mahindra": "M&M",
    "M&M": "M&M",
    "Mahindra": "M&M",
    # --- Bajaj Auto ---
    "Bajaj Auto": "BAJAJ-AUTO",
    "BAJAJ-AUTO": "BAJAJ-AUTO",
    # --- TVS Motor ---
    "TVS Motor Company": "TVSMOTOR",
    "TVS Motor": "TVSMOTOR",
    "TVSMOTOR": "TVSMOTOR",
    # --- Motherson ---
    "Motherson Sumi": "MOTHERSON",
    "Motherson": "MOTHERSON",
    "MOTHERSON": "MOTHERSON",
    # --- BHEL ---
    "Bharat Heavy Electricals": "BHEL",
    "BHEL": "BHEL",
    # --- BEL ---
    "Bharat Electronics": "BEL",
    "BEL": "BEL",
    # --- HAL ---
    "Hindustan Aeronautics": "HAL",
    "HAL": "HAL",
    # --- IRCTC ---
    "IRCTC": "IRCTC",
    # --- LIC ---
    "Life Insurance Corporation": "LICI",
    "LIC": "LICI",
    "LICI": "LICI",
    # --- Polycab ---
    "Polycab India": "POLYCAB",
    "Polycab": "POLYCAB",
    "POLYCAB": "POLYCAB",
    # --- ABB ---
    "ABB India": "ABB",
    "ABB": "ABB",
    # --- Trent ---
    "Trent Limited": "TRENT",
    "Trent": "TRENT",
    "TRENT": "TRENT",
    # --- Persistent ---
    "Persistent Systems": "PERSISTENT",
    "Persistent": "PERSISTENT",
    "PERSISTENT": "PERSISTENT",
    # --- Coforge ---
    "Coforge": "COFORGE",
    "COFORGE": "COFORGE",
    # --- LTIMindtree ---
    "LTIMindtree": "LTIM",
    "LTI Mindtree": "LTIM",
    "LTIM": "LTIM",
    # --- Mphasis ---
    "Mphasis": "MPHASIS",
    "MPHASIS": "MPHASIS",
    # --- Naukri / Info Edge ---
    "Info Edge": "NAUKRI",
    "InfoEdge": "NAUKRI",
    "Naukri": "NAUKRI",
    "NAUKRI": "NAUKRI",
    # --- Zomato ---
    "Zomato": "ZOMATO",
    "ZOMATO": "ZOMATO",
    # --- Paytm / One97 ---
    "Paytm": "PAYTM",
    "One97 Communications": "PAYTM",
    "One97": "PAYTM",
    "PAYTM": "PAYTM",
    # --- DMart / Avenue ---
    "Avenue Supermarts": "DMART",
    "DMart": "DMART",
    "DMART": "DMART",
    # --- Page Industries ---
    "Page Industries": "PAGEIND",
    "PAGEIND": "PAGEIND",
    # --- Cholamandalam ---
    "Cholamandalam Investment": "CHOLAFIN",
    "Cholamandalam": "CHOLAFIN",
    "CHOLAFIN": "CHOLAFIN",
    # --- Shriram Finance ---
    "Shriram Finance": "SHRIRAMFIN",
    "Shriram Transport": "SHRIRAMFIN",
    "SHRIRAMFIN": "SHRIRAMFIN",
    # --- M&M Finance ---
    "Mahindra Finance": "M&MFIN",
    "M&MFIN": "M&MFIN",
    "M&M Finance": "M&MFIN",
    # --- PEL ---
    "Piramal Enterprises": "PEL",
    "Piramal": "PEL",
    "PEL": "PEL",
    # --- MFSL ---
    "Max Financial Services": "MFSL",
    "Max Financial": "MFSL",
    "MFSL": "MFSL",
    # --- Manappuram ---
    "Manappuram Finance": "MANAPPURAM",
    "Manappuram": "MANAPPURAM",
    "MANAPPURAM": "MANAPPURAM",
    # --- Muthoot ---
    "Muthoot Finance": "MUTHOOTFIN",
    "Muthoot": "MUTHOOTFIN",
    "MUTHOOTFIN": "MUTHOOTFIN",
    # --- Cummins ---
    "Cummins India": "CUMMINSIND",
    "Cummins": "CUMMINSIND",
    "CUMMINSIND": "CUMMINSIND",
    # --- Voltas ---
    "Voltas": "VOLTAS",
    "VOLTAS": "VOLTAS",
    # --- Crompton ---
    "Crompton Greaves Consumer": "CROMPTON",
    "Crompton": "CROMPTON",
    "CROMPTON": "CROMPTON",
    # --- Whirlpool ---
    "Whirlpool India": "WHIRLPOOL",
    "Whirlpool": "WHIRLPOOL",
    "WHIRLPOOL": "WHIRLPOOL",
    # --- Bata ---
    "Bata India": "BATAINDIA",
    "Bata": "BATAINDIA",
    "BATAINDIA": "BATAINDIA",
    # --- Tata Power ---
    "Tata Power": "TATAPOWER",
    "TATAPOWER": "TATAPOWER",
    # --- Adani Green ---
    "Adani Green Energy": "ADANIGREEN",
    "Adani Green": "ADANIGREEN",
    "ADANIGREEN": "ADANIGREEN",
    # --- Torrent Pharma ---
    "Torrent Pharmaceuticals": "TORNTPHARM",
    "Torrent Pharma": "TORNTPHARM",
    "TORNTPHARM": "TORNTPHARM",
    # --- Aurobindo ---
    "Aurobindo Pharma": "AUROPHARMA",
    "Aurobindo": "AUROPHARMA",
    "AUROPHARMA": "AUROPHARMA",
    # --- Biocon ---
    "Biocon": "BIOCON",
    "BIOCON": "BIOCON",
    # --- Lupin ---
    "Lupin": "LUPIN",
    "LUPIN": "LUPIN",
    # --- Alkem ---
    "Alkem Laboratories": "ALKEM",
    "Alkem": "ALKEM",
    "ALKEM": "ALKEM",
    # --- Laurus Labs ---
    "Laurus Labs": "LAURUSLABS",
    "LAURUSLABS": "LAURUSLABS",
    # --- IPCA Labs ---
    "IPCA Laboratories": "IPCALAB",
    "IPCA Labs": "IPCALAB",
    "IPCALAB": "IPCALAB",
    # --- IndiGo ---
    "InterGlobe Aviation": "INDIGO",
    "IndiGo": "INDIGO",
    "Indigo Airlines": "INDIGO",
    "INDIGO": "INDIGO",
    # --- CONCOR ---
    "Container Corporation": "CONCOR",
    "CONCOR": "CONCOR",
    # --- SRF ---
    "SRF Limited": "SRF",
    "SRF": "SRF",
    # --- PI Industries ---
    "PI Industries": "PIIND",
    "PIIND": "PIIND",
    # --- Atul ---
    "Atul Limited": "ATUL",
    "Atul": "ATUL",
    "ATUL": "ATUL",
    # --- Deepak Nitrite ---
    "Deepak Nitrite": "DEEPAKNTR",
    "DEEPAKNTR": "DEEPAKNTR",
    # --- Coromandel ---
    "Coromandel International": "COROMANDEL",
    "Coromandel": "COROMANDEL",
    "COROMANDEL": "COROMANDEL",
    # --- UBL ---
    "United Breweries": "UBL",
    "UBL": "UBL",
    # --- Colgate ---
    "Colgate-Palmolive India": "COLPAL",
    "Colgate Palmolive": "COLPAL",
    "Colgate": "COLPAL",
    "COLPAL": "COLPAL",
    # --- Marico ---
    "Marico": "MARICO",
    "MARICO": "MARICO",
    # --- Berger Paints ---
    "Berger Paints": "BERGEPAINT",
    "Berger": "BERGEPAINT",
    "BERGEPAINT": "BERGEPAINT",
    # --- Balkrishna ---
    "Balkrishna Industries": "BALKRISIND",
    "Balkrishna": "BALKRISIND",
    "BALKRISIND": "BALKRISIND",
    # --- MRF ---
    "MRF": "MRF",
    # --- Exide ---
    "Exide Industries": "EXIDEIND",
    "Exide": "EXIDEIND",
    "EXIDEIND": "EXIDEIND",
    # --- Amara Raja ---
    "Amara Raja Batteries": "AMARAJABAT",
    "Amara Raja": "AMARAJABAT",
    "AMARAJABAT": "AMARAJABAT",
    # --- Ashok Leyland ---
    "Ashok Leyland": "ASHOKLEY",
    "ASHOKLEY": "ASHOKLEY",
    # --- Escorts ---
    "Escorts Kubota": "ESCORTS",
    "Escorts": "ESCORTS",
    "ESCORTS": "ESCORTS",
    # --- LIC Housing ---
    "LIC Housing Finance": "LICHSGFIN",
    "LIC Housing": "LICHSGFIN",
    "LICHSGFIN": "LICHSGFIN",
    # --- REC ---
    "REC Limited": "RECLTD",
    "REC": "RECLTD",
    "RECLTD": "RECLTD",
    # --- PFC ---
    "Power Finance Corporation": "PFC",
    "PFC": "PFC",
    # --- IRFC ---
    "Indian Railway Finance": "IRFC",
    "IRFC": "IRFC",
    # --- SAIL ---
    "Steel Authority of India": "SAIL",
    "SAIL": "SAIL",
    # --- Hindustan Zinc ---
    "Hindustan Zinc": "HINDZINC",
    "HINDZINC": "HINDZINC",
    # --- NMDC ---
    "NMDC": "NMDC",
    # --- Vedanta ---
    "Vedanta": "VEDL",
    "VEDL": "VEDL",
    # --- GAIL ---
    "GAIL India": "GAIL",
    "GAIL": "GAIL",
    # --- IOC ---
    "Indian Oil Corporation": "IOC",
    "Indian Oil": "IOC",
    "IOC": "IOC",
    # --- Petronet LNG ---
    "Petronet LNG": "PETRONET",
    "Petronet": "PETRONET",
    "PETRONET": "PETRONET",
    # --- Indus Towers ---
    "Indus Towers": "INDUSTOWER",
    "INDUSTOWER": "INDUSTOWER",
    # --- Vodafone Idea ---
    "Vodafone Idea": "IDEA",
    "Vi": "IDEA",
    "IDEA": "IDEA",
    # --- Zee ---
    "Zee Entertainment": "ZEEL",
    "ZEEL": "ZEEL",
    # --- PVR Inox ---
    "PVR Inox": "PVRINOX",
    "PVRINOX": "PVRINOX",
    "PVR": "PVRINOX",
    # --- Can Fin Homes ---
    "Can Fin Homes": "CANFINHOME",
    "CANFINHOME": "CANFINHOME",
    # --- Oberoi Realty ---
    "Oberoi Realty": "OBEROIRLTY",
    "OBEROIRLTY": "OBEROIRLTY",
    "Oberoi": "OBEROIRLTY",
    # --- Godrej Properties ---
    "Godrej Properties": "GODREJPROP",
    "GODREJPROP": "GODREJPROP",
    # --- Prestige Estates ---
    "Prestige Estates": "PRESTIGE",
    "Prestige": "PRESTIGE",
    "PRESTIGE": "PRESTIGE",
    # --- Ramco Cements ---
    "Ramco Cements": "RAMCOCEM",
    "RAMCOCEM": "RAMCOCEM",
    # --- Shree Cement ---
    "Shree Cement": "SHREECEM",
    "SHREECEM": "SHREECEM",
}

# Validate all aliases map to F&O stocks
_FNO_SET = set(FNO_UNIVERSE.keys())
for _alias, _sym in COMPANY_ALIASES.items():
    assert _sym in _FNO_SET, f"Alias {_alias!r} → {_sym!r} not in FNO_UNIVERSE"


# ---------------------------------------------------------------------------
# Index keywords → index name
# ---------------------------------------------------------------------------

INDEX_KEYWORDS: dict[str, str] = {
    "NIFTY 50": "NIFTY",
    "NIFTY50": "NIFTY",
    "Nifty": "NIFTY",
    "SENSEX": "NIFTY",
    "BSE 30": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "Bank Nifty": "BANKNIFTY",
    "banking sector": "BANKNIFTY",
    "RBI": "BANKNIFTY",
    "rate cut": "BANKNIFTY",
    "rate hike": "BANKNIFTY",
    "repo rate": "BANKNIFTY",
    "monetary policy": "BANKNIFTY",
    "SEBI": "NIFTY",
    "GDP": "NIFTY",
    "fiscal deficit": "NIFTY",
    "Union Budget": "NIFTY",
    "budget": "NIFTY",
}

# ---------------------------------------------------------------------------
# Short aliases that need word-boundary matching (3 chars or less)
# to avoid false positives (e.g., "IT" in "with", "DLF" in text).
# Longer aliases are matched with simple substring (case-insensitive).
# ---------------------------------------------------------------------------

_SHORT_THRESHOLD = 4  # aliases < this length require word-boundary match

# Pre-sort aliases by length descending → longest match first
_SORTED_ALIASES: list[tuple[str, str]] = sorted(
    COMPANY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True
)

# Pre-compile word-boundary patterns for short aliases
_SHORT_PATTERNS: dict[str, re.Pattern] = {}
for _alias, _ in _SORTED_ALIASES:
    if len(_alias) < _SHORT_THRESHOLD:
        _SHORT_PATTERNS[_alias] = re.compile(
            r"(?<![A-Za-z])" + re.escape(_alias) + r"(?![A-Za-z])",
            re.IGNORECASE,
        )


def extract_stocks(headline: str) -> list[str]:
    """Extract F&O stock symbols mentioned in a headline.

    Uses longest-match-first to avoid partial matches
    (e.g., "Tata Consultancy" matches TCS, not "Tata" → TATAMOTORS).

    Returns sorted deduplicated list of NSE symbols.
    """
    found: set[str] = set()
    remaining = headline

    for alias, symbol in _SORTED_ALIASES:
        if len(alias) < _SHORT_THRESHOLD:
            pattern = _SHORT_PATTERNS[alias]
            if pattern.search(remaining):
                found.add(symbol)
                # Remove matched portions to prevent shorter aliases matching
                remaining = pattern.sub("", remaining)
        else:
            # Case-insensitive substring match for longer aliases
            if alias.lower() in remaining.lower():
                found.add(symbol)
                # Remove matched text (case-insensitive)
                remaining = re.sub(re.escape(alias), "", remaining, flags=re.IGNORECASE)

    return sorted(found)


def extract_indices(headline: str) -> list[str]:
    """Extract index references from a headline.

    Returns deduplicated list of index names (e.g., ["NIFTY", "BANKNIFTY"]).
    """
    found: set[str] = set()
    hl_lower = headline.lower()
    for keyword, index in INDEX_KEYWORDS.items():
        if keyword.lower() in hl_lower:
            found.add(index)
    return sorted(found)
