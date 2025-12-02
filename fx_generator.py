# fx_generator.py

import pandas as pd  # type: ignore
import numpy as np   # type: ignore
import random
from datetime import datetime, timedelta, date
import uuid
import re
from math import floor
from typing import List, Dict
from pydantic import BaseModel # type: ignore

# ---------------- CONFIG MODELS (used by FastAPI & generator) ----------------

class AmountBin(BaseModel):
    lo: int
    hi: int

class FXConfig(BaseModel):
    n_rows: int = 100
    random_seed: int = 42

    banks: List[str] = ["10001", "20002"]

    date_start: date = date(2025, 7, 1)
    date_span_days: int = 120
    timeperiod_bucket_minutes: int = 30

    currencies: List[str] = ["USD", "EUR", "RUB", "AMD", "GBP"]
    base_to_amd: Dict[str, float] = {
        "USD": 400.0,
        "EUR": 430.0,
        "RUB": 4.5,
        "AMD": 1.0,
        "GBP": 500.0,
    }

    amount_bins: List[AmountBin] = [
        AmountBin(lo=0, hi=100_000),
        AmountBin(lo=100_000, hi=400_000),
        AmountBin(lo=400_000, hi=1_500_000),
        AmountBin(lo=1_500_000, hi=3_000_000),
        AmountBin(lo=3_000_000, hi=6_000_000),
        AmountBin(lo=6_000_000, hi=11_000_000),
        AmountBin(lo=11_000_000, hi=20_000_000),
        AmountBin(lo=20_000_000, hi=40_000_000),
        AmountBin(lo=40_000_000, hi=80_000_000),
        AmountBin(lo=80_000_000, hi=220_000_000),
        AmountBin(lo=220_000_000, hi=800_000_000),
    ]

# ---------------- STATIC CONSTANTS (not exposed for now) ----------------

ALLOWED_FORMS = {"Փոխանցում", "Վճարում", "Հաշվի համալրում", "Հաշվից կանխիկացում"}
COUNTRIES = ["AM", "US", "RU", "DE", "FR", "AE", "CN", "GB", "GE"]
RESIDENCY = ["Resident", "Nonresident"]
LEGAL_STATUS = ["Nat", "Leg", "SEnt", "NA"]
LINK_FLAG = ["Linked", "Nonlinked"]
EXEC_METHOD = ["Cash", "Non cash"]  # exact tokens
ON_WHOSE_BEHALF = ["OnItsBehalf", "OnCustBehalf"]
TX_TYPE = ["Direct", "Indirect"]
ENV_OPTIONS = ["InPerson", "Web", "Mobile", "FGIAS", "POS", "VirtualPOS"]
SIGNING_PLACE_EXAMPLES = [
    "Yerevan Office-1", "Yerevan Office-2",
    "Gyumri Branch-1", "Vanadzor-Desk"
]


# ---------------- MAIN GENERATOR FUNCTION ----------------

def generate_fx_details(config: FXConfig) -> pd.DataFrame:
    # --- derive variables from config ---
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    BANKS = [{"organization_id": org} for org in config.banks]
    DATE_START = datetime.combine(config.date_start, datetime.min.time())
    DATE_SPAN_DAYS = config.date_span_days
    TIMEPERIOD_BUCKET_MINUTES = config.timeperiod_bucket_minutes

    CURRENCIES = config.currencies
    BASE_TO_AMD = config.base_to_amd
    AMOUNT_BINS = [(b.lo, b.hi) for b in config.amount_bins]

    # --- helper functions (same logic as your script, but using these vars) ---

    def make_parent_message_id(org_code: str, dt: datetime, seq: int) -> str:
        return f"{org_code}{dt.strftime('%d%m%y')}{seq:02d}"

    def make_external_id(org_code: str) -> str:
        return f"{org_code}-{uuid.uuid4()}"

    def sample_fx_pair():
        if random.random() < 0.8:
            sell = "AMD"
            buy = random.choice([c for c in CURRENCIES if c != "AMD"])
        else:
            buy, sell = random.sample(CURRENCIES, 2)
        return buy, sell

    def compute_rate(buy: str, sell: str) -> float:
        rate = BASE_TO_AMD[buy] / BASE_TO_AMD[sell]
        return max(1e-9, round(rate * np.random.normal(1.0, 0.002), 6))

    def pick_amount_bin(amd_val: float):
        for lo, hi in AMOUNT_BINS:
            if lo <= amd_val < hi:
                return lo, hi
        return AMOUNT_BINS[-1]

    def choose_on_whose():
        on_whose = random.choice(ON_WHOSE_BEHALF)
        return on_whose, ("OnItsName" if on_whose == "OnItsBehalf" else "OnCustName")

    def to_timeperiod_bucket_with_seconds(dt: datetime) -> str:
        start_min = (dt.minute // TIMEPERIOD_BUCKET_MINUTES) * TIMEPERIOD_BUCKET_MINUTES
        start = dt.replace(minute=start_min, second=0, microsecond=0)
        end = start + timedelta(minutes=TIMEPERIOD_BUCKET_MINUTES)
        return f"{start.strftime('%H:%M:%S')} – {end.strftime('%H:%M:%S')}"

    def gen_psn_or_certificate(p_has_psn=0.85) -> str:
        if random.random() < p_has_psn:
            return f"{random.randint(1_000_000_000, 9_999_999_999)}"
        else:
            return f"CERT-{random.randint(1_000_000, 99_999_999)}"

    def is_valid_partner_ssn(val) -> bool:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return True
        return bool(re.fullmatch(r"(\d{10}|CERT-\d{7,8})", str(val)))

    def gen_tin() -> str:
        return str(random.randint(10_000_000, 99_999_999))

    def is_cash_pair(buy_exec: str, sell_exec: str) -> bool:
        return (buy_exec == "Cash") and (sell_exec == "Cash")

    def pick_exec_environment(buy_exec: str, sell_exec: str, tx_type: str) -> str:
        if is_cash_pair(buy_exec, sell_exec):
            allowed = ["InPerson", "FGIAS"]
        else:
            allowed = (
                ["Web", "Mobile", "POS", "InPerson"]
                if tx_type == "Direct"
                else ["Web", "Mobile", "POS", "VirtualPOS"]
            )
        return random.choice(allowed)

    def round2(x: float) -> float:
        return float(f"{x:.2f}")

    def integer_digits_len(x: float) -> int:
        return len(str(int(floor(abs(x)))))

    # ---------------- MAIN LOOP (mostly your original logic) ----------------

    rows = []
    seq_map = {b["organization_id"]: 1 for b in BANKS}

    for _ in range(config.n_rows):
        bank = random.choice(BANKS)
        org = bank["organization_id"]

        dt = DATE_START + timedelta(
            days=random.randint(0, DATE_SPAN_DAYS),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        signing_date = dt.strftime("%d/%m/%Y")
        signing_time = dt.strftime("%H:%M:%S")
        exec_dt = dt + timedelta(minutes=random.randint(0, 120))
        execution_date = exec_dt.strftime("%d/%m/%Y")
        execution_time = exec_dt.strftime("%H:%M:%S")
        time_period = to_timeperiod_bucket_with_seconds(dt)

        parent_message_id = make_parent_message_id(org, dt, seq_map[org])
        seq_map[org] += 1
        external_id = make_external_id(org)

        buy_cur, sell_cur = sample_fx_pair()
        ex_rate = compute_rate(buy_cur, sell_cur)
        pub_rate = ex_rate if random.random() < 0.7 else None

        base_vol = abs(np.random.normal(1000, 500)) + 50
        internal_buy_vol = round2(base_vol)
        internal_sell_vol = round2(internal_buy_vol * ex_rate)

        buy_in_amd = internal_buy_vol * BASE_TO_AMD[buy_cur]
        sell_in_amd = internal_sell_vol * BASE_TO_AMD[sell_cur]
        amd_amount = float(max(buy_in_amd, sell_in_amd))
        lo, hi = pick_amount_bin(amd_amount)
        amount_range = f"{int(lo)} – {int(hi)}"

        account_on_whose, name_on_whose = choose_on_whose()
        tx_type = "Direct" if random.random() < 0.85 else "Indirect"
        tx_form = random.choice(list(ALLOWED_FORMS)) if tx_type == "Indirect" else None

        # Partner classifiers (ALWAYS populated)
        partner_country = random.choice(COUNTRIES)
        partner_residency = "Resident" if partner_country == "AM" else random.choice(RESIDENCY)
        partner_legal = random.choice(LEGAL_STATUS)
        partner_link = random.choice(LINK_FLAG)
        established = (partner_link == "Linked")
        partner_code = (
            f"CBA-{random.randint(100000, 999999)}"
            if partner_legal in ["Leg", "SEnt"] and random.random() < 0.5
            else None
        )

        commission = round2(abs(np.random.normal(1500, 800))) if random.random() < 0.35 else None
        include_client = (account_on_whose == "OnCustBehalf")
        include_intermediary = random.random() < 0.2

        # Partner identity
        partner_ssn = partner_passport = partner_tin = partner_legalName = None
        if partner_legal == "Nat":
            if random.random() < 0.9:
                partner_ssn = gen_psn_or_certificate(0.8)
            if partner_ssn is None:
                if (amd_amount >= 100_000 and tx_type == "Direct" and established) or (
                    account_on_whose == "OnCustBehalf"
                ):
                    partner_passport = f"AP{random.randint(100000, 999999)}"
                elif random.random() < 0.5:
                    partner_passport = f"AP{random.randint(100000, 999999)}"
        else:
            is_leglike = partner_legal in ["Leg", "SEnt", "NA"]
            if is_leglike:
                if (
                    (partner_residency == "Resident" and amd_amount >= 100_000 and tx_type == "Direct" and established)
                    or (account_on_whose == "OnCustBehalf" and partner_residency == "Resident")
                ):
                    partner_tin = gen_tin()
                elif random.random() < 0.5:
                    partner_tin = gen_tin()
                need_name1 = (partner_tin is None) and (amd_amount >= 100_000) and (tx_type != "Direct") and established
                need_name2 = (partner_tin is None) and (account_on_whose == "OnCustBehalf")
                if need_name1 or need_name2:
                    partner_legalName = f"Ընկերություն {random.randint(1000, 9999)}"
                elif random.random() < 0.15:
                    partner_legalName = f"Ընկերություն {random.randint(1000, 9999)}"

        # Client (visible only OnCustBehalf)
        client_country = client_residency = client_legal = client_link = client_code = None
        client_ssn = client_passport = client_tin = client_legalName = None
        if include_client:
            client_country = random.choice(COUNTRIES)
            client_residency = "Resident" if client_country == "AM" else random.choice(RESIDENCY)
            client_legal = random.choice(LEGAL_STATUS)
            client_link = random.choice(LINK_FLAG)
            if client_legal == "Nat":
                client_ssn = f"{random.randint(1_000_000_000, 9_999_999_999)}"
            if client_legal in ["Leg", "SEnt", "NA"]:
                if client_residency == "Resident":
                    client_tin = gen_tin()
                else:
                    if random.random() < 0.5:
                        client_tin = gen_tin()
                    if client_tin is None:
                        client_legalName = f"Պատվիրատու ՍՊԸ {random.randint(100, 999)}"
                if client_tin is None and client_legalName is None:
                    client_legalName = f"Պատվիրատու ՍՊԸ {random.randint(100, 999)}"

        # Intermediary
        intermediary_country = intermediary_residency = intermediary_link = intermediary_code = None
        intermediary_tin = intermediary_legalName = None
        if include_intermediary:
            intermediary_country = random.choice(COUNTRIES)
            intermediary_residency = (
                "Resident" if intermediary_country == "AM" else random.choice(RESIDENCY)
            )
            intermediary_link = random.choice(LINK_FLAG)
            if random.random() < 0.25:
                intermediary_code = f"CBA-{random.randint(100000, 999999)}"
            if random.random() < 0.6:
                intermediary_tin = gen_tin()
            if intermediary_tin is None:
                intermediary_legalName = f"Միջնորդ {random.randint(1000, 9999)}"

        buy_exec = random.choice(EXEC_METHOD)
        sell_exec = random.choice(EXEC_METHOD)
        exec_env = pick_exec_environment(buy_exec, sell_exec, tx_type)
        signing_place = random.choice(SIGNING_PLACE_EXAMPLES)
        sectoral = random.choice(["Financial", "Public", "Private", "Household", "NA"])
        econ_branch = random.choice(["B_01", "B_02", "B_15", "B_21", "NA"])

        # expose exactly one volume (XOR)
        if random.random() < 0.5:
            buy_vol, sell_vol = internal_buy_vol, None
        else:
            buy_vol, sell_vol = None, internal_sell_vol

        rows.append({
            "external_id": external_id,
            "parent_message_id": parent_message_id,
            "organization_id": org,
            "branchId": f"{random.randint(1, 3):03d}",
            "partner_country": partner_country,
            "partner_residency": partner_residency,
            "partner_legalStatus": partner_legal,
            "partner_sectOralAffiliation": sectoral,
            "partner_economicBranch": econ_branch,
            "partner_codeByCBA": partner_code,
            "partner_linkToFinancialInstitution": partner_link,
            "partner_ssn": partner_ssn,
            "partner_passport": partner_passport,
            "partner_tin": partner_tin,
            "partner_legalName": partner_legalName,
            "client_country": client_country,
            "client_residency": client_residency,
            "client_legalStatus": client_legal,
            "client_sectoralAffiliation": (
                random.choice(["Financial", "Public", "Private", "Household", "NA"])
                if include_client
                else None
            ),
            "client_economicBranch": (
                random.choice(["G_46", "G_47", "M_70", "NA"])
                if include_client
                else None
            ),
            "client_codeByCBA": client_code,
            "client_linkToFinancialInstitution": client_link,
            "client_ssn": client_ssn,
            "client_passport": client_passport,
            "client_tin": client_tin,
            "client_legalName": client_legalName,
            "intermediary_country": intermediary_country,
            "intermediary_residency": intermediary_residency,
            "intermediary_codeByCBA": intermediary_code,
            "intermediary_linkToFinancialInstitution": intermediary_link,
            "intermediary_tin": intermediary_tin,
            "intermediary_legalName": intermediary_legalName,
            "buyCurrency": buy_cur,
            "sellCurrency": sell_cur,
            "buyVolume": buy_vol,
            "sellVolume": sell_vol,
            "exchangeRate": ex_rate,
            "publishedExchangeRate": pub_rate,
            "amountRange": amount_range,
            "buyExecutionMethod": buy_exec,
            "sellExecutionMethod": sell_exec,
            "commissionFee_AMD": commission,
            "accountOnWhoseBehalf": account_on_whose,
            "nameOnWhoseBehalf": name_on_whose,
            "transactionType": tx_type,
            "transactionForm": tx_form,
            "transactionSigningDate": signing_date,
            "transactionExecutionDate": execution_date,
            "SigningTime": signing_time,
            "ExecutionTime": execution_time,
            "timePeriod": time_period,
            "transactionSigningPlace": signing_place,
            "transactionExecutionEnvironment": exec_env,
            "_amdAmount_AMD": amd_amount,
            "_internal_buy_vol": internal_buy_vol,
            "_internal_sell_vol": internal_sell_vol,
        })

    df = pd.DataFrame(rows)

    # ---------------- GROUP-UNIFORM SETUP (unchanged logic) ----------------

    NUM_SUMMARY_GROUPS = 5
    idx_groups = np.array_split(np.arange(len(df)), NUM_SUMMARY_GROUPS)

    COLS_BRANCH = ["branchId"]
    COLS_PARTNER_7_13 = [
        "partner_ssn", "partner_passport", "partner_tin", "partner_legalName",
        "partner_country", "partner_residency", "partner_legalStatus",
    ]
    COLS_CLIENT_18_24 = [
        "client_ssn", "client_passport", "client_tin", "client_legalName",
        "client_country", "client_residency", "client_legalStatus",
    ]
    COLS_INTERMEDIARY_27_32 = [
        "intermediary_tin", "intermediary_legalName", "intermediary_country",
        "intermediary_residency", "intermediary_codeByCBA",
        "intermediary_linkToFinancialInstitution",
    ]
    COLS_RATES_35_39 = [
        "buyCurrency", "sellCurrency", "exchangeRate", "publishedExchangeRate",
        "amountRange",
    ]
    COLS_METHODS_41_43 = [
        "buyExecutionMethod", "sellExecutionMethod", "accountOnWhoseBehalf",
    ]
    COLS_NAMES_45_46 = [
        "nameOnWhoseBehalf", "transactionType",
    ]
    COLS_ENV_48_50 = [
        "timePeriod", "transactionSigningPlace", "transactionExecutionEnvironment",
    ]
    COLS_TO_LOCK = (
        COLS_BRANCH
        + COLS_PARTNER_7_13
        + COLS_CLIENT_18_24
        + COLS_INTERMEDIARY_27_32
        + COLS_RATES_35_39
        + COLS_METHODS_41_43
        + COLS_NAMES_45_46
        + COLS_ENV_48_50
    )

    def _env_ok(buy_e: str, sell_e: str, tx_type: str, env: str) -> bool:
        if (buy_e == "Cash") and (sell_e == "Cash"):
            return env in {"InPerson", "FGIAS"}
        return env in (
            {"Web", "Mobile", "POS", "InPerson"} if tx_type == "Direct"
            else {"Web", "Mobile", "POS", "VirtualPOS"}
        )

    for gidxs in idx_groups:
        tpl = df.iloc[gidxs[0]].copy()
        be, se, ttype = tpl["buyExecutionMethod"], tpl["sellExecutionMethod"], tpl["transactionType"]
        if not _env_ok(be, se, ttype, tpl["transactionExecutionEnvironment"]):
            tpl["transactionExecutionEnvironment"] = pick_exec_environment(be, se, ttype)

        if tpl["accountOnWhoseBehalf"] == "OnItsBehalf":
            for c in [
                "client_country","client_residency","client_legalStatus",
                "client_sectoralAffiliation","client_economicBranch",
                "client_codeByCBA","client_linkToFinancialInstitution",
                "client_ssn","client_passport","client_tin","client_legalName"
            ]:
                if c in df.columns:
                    tpl[c] = None

        for c in COLS_TO_LOCK:
            if c in df.columns:
                df.loc[gidxs, c] = tpl[c]

    # ---------------- POST-ENFORCEMENT + ASSERTIONS (unchanged) ----------------

    mask_amount_100k = df["_amdAmount_AMD"] >= 100_000
    mask_direct_tx = df["transactionType"] == "Direct"
    mask_established = df["partner_linkToFinancialInstitution"] == "Linked"
    mask_partner_mand = mask_amount_100k & mask_direct_tx & mask_established

    mask_plegal_present = df["partner_legalStatus"].notna()
    mask_nat = df["partner_legalStatus"] == "Nat"
    mask_leglike = df["partner_legalStatus"].isin(["Leg", "SEnt", "NA"])

    df.loc[mask_plegal_present & mask_nat, "partner_sectOralAffiliation"] = "Household"
    df.loc[~df["partner_legalStatus"].isin(["Leg", "SEnt"]), "partner_codeByCBA"] = None

    df.loc[mask_plegal_present & ~mask_nat, "partner_ssn"] = None
    assert df.loc[mask_plegal_present, "partner_ssn"].apply(is_valid_partner_ssn).all()
    df.loc[mask_plegal_present & ~mask_nat, "partner_passport"] = None
    df.loc[mask_plegal_present & ~mask_leglike, "partner_tin"] = None
    df.loc[mask_plegal_present & mask_nat, "partner_legalName"] = None

    mask_oncust = df["accountOnWhoseBehalf"] == "OnCustBehalf"
    client_presence_cols = [
        "client_country","client_residency","client_legalStatus",
        "client_sectoralAffiliation","client_economicBranch",
        "client_codeByCBA","client_linkToFinancialInstitution"
    ]
    for c in client_presence_cols:
        df.loc[~mask_oncust, c] = None

    def fill_if_null(series, filler):
        m = series.isna() & mask_oncust
        if m.any():
            series.loc[m] = [filler() for _ in range(m.sum())]
        return series

    df["client_country"] = fill_if_null(df["client_country"], lambda: random.choice(COUNTRIES))
    df["client_residency"] = fill_if_null(df["client_residency"], lambda: random.choice(RESIDENCY))
    df["client_legalStatus"] = fill_if_null(df["client_legalStatus"], lambda: random.choice(LEGAL_STATUS))

    mask_client_nat = df["client_legalStatus"] == "Nat"
    df.loc[mask_oncust & mask_client_nat, "client_ssn"] = df.loc[
        mask_oncust & mask_client_nat, "client_ssn"
    ].where(
        df.loc[mask_oncust & mask_client_nat, "client_ssn"].notna(),
        [f"{random.randint(1_000_000_000, 9_999_999_999)}"
         for _ in range((mask_oncust & mask_client_nat).sum())]
    )
    df.loc[~(mask_oncust & mask_client_nat), "client_ssn"] = None

    need_client_passport = mask_oncust & mask_client_nat & df["client_ssn"].isna()
    df.loc[need_client_passport, "client_passport"] = df.loc[
        need_client_passport, "client_passport"
    ].where(
        df.loc[need_client_passport, "client_passport"].notna(),
        [f"AP{random.randint(100000, 999999)}"
         for _ in range(need_client_passport.sum())]
    )

    mask_leglike_client = df["client_legalStatus"].isin(["Leg", "SEnt", "NA"])
    mask_client_resident = df["client_residency"] == "Resident"
    need_client_tin = mask_oncust & mask_leglike_client & mask_client_resident

    df.loc[need_client_tin, "client_tin"] = df.loc[
        need_client_tin, "client_tin"
    ].where(
        df.loc[need_client_tin, "client_tin"].notna(),
        [str(random.randint(10_000_000, 99_999_999))
         for _ in range(need_client_tin.sum())]
    )
    df.loc[mask_oncust & ~mask_leglike_client, "client_tin"] = None

    mask_client_tin_null = df["client_tin"].isna()
    need_client_name = mask_oncust & mask_leglike_client & mask_client_tin_null
    df.loc[need_client_name, "client_legalName"] = df.loc[
        need_client_name, "client_legalName"
    ].where(
        df.loc[need_client_name, "client_legalName"].notna(),
        [f"Պատվիրատու ՍՊԸ {random.randint(100, 999)}"
         for _ in range(need_client_name.sum())]
    )
    df.loc[mask_oncust & ~mask_leglike_client, "client_legalName"] = None

    id_or_link_cols = [
        "intermediary_tin","intermediary_legalName",
        "intermediary_codeByCBA","intermediary_linkToFinancialInstitution"
    ]
    mask_i_present = df["intermediary_country"].notna() & df["intermediary_residency"].notna()
    df.loc[~mask_i_present, id_or_link_cols] = None

    mask_i_need_name = mask_i_present & df["intermediary_tin"].isna()
    df.loc[mask_i_need_name, "intermediary_legalName"] = df.loc[
        mask_i_need_name, "intermediary_legalName"
    ].where(
        df.loc[mask_i_need_name, "intermediary_legalName"].notna(),
        [f"Միջնորդ {random.randint(1000, 9999)}"
         for _ in range(mask_i_need_name.sum())]
    )

    mask_indirect = df["transactionType"] == "Indirect"
    mask_direct_only = df["transactionType"] == "Direct"
    df.loc[mask_direct_only, "transactionForm"] = None
    df.loc[mask_indirect, "transactionForm"] = df.loc[
        mask_indirect, "transactionForm"
    ].fillna("Փոխանցում")

    def env_allowed(row) -> bool:
        buy_e, sell_e = row["buyExecutionMethod"], row["sellExecutionMethod"]
        env = row["transactionExecutionEnvironment"]
        if is_cash_pair(buy_e, sell_e):
            return env in {"InPerson", "FGIAS"}
        return env in (
            {"Web","Mobile","POS","InPerson"}
            if row["transactionType"] == "Direct"
            else {"Web","Mobile","POS","VirtualPOS"}
        )

    bad_env_mask = ~df.apply(env_allowed, axis=1)
    df.loc[bad_env_mask, "transactionExecutionEnvironment"] = df[bad_env_mask].apply(
        lambda r: pick_exec_environment(
            r["buyExecutionMethod"],
            r["sellExecutionMethod"],
            r["transactionType"]
        ),
        axis=1
    )

    xor_ok = (df["buyVolume"].notna() ^ df["sellVolume"].notna())
    assert xor_ok.all(), "Exactly one of buyVolume/sellVolume must be filled"

    def check_volume_rules(series: pd.Series, name: str):
        s = series.dropna()
        if s.empty:
            return
        assert (s > 0).all(), f"{name} must be > 0"
        assert s.apply(integer_digits_len).le(12).all(), f"{name} integer part must have ≤ 12 digits"
        assert s.apply(
            lambda v: re.fullmatch(r"^\d{1,12}(\.\d{1,2})?$", f"{v:.2f}")
        ).all(), f"{name} must have ≤ 2 decimals"

    check_volume_rules(df["buyVolume"], "buyVolume")
    check_volume_rules(df["sellVolume"], "sellVolume")

    assert (df["exchangeRate"] > 0).all()
    if df["publishedExchangeRate"].notna().any():
        assert (df["publishedExchangeRate"].dropna() > 0).all()

    assert set(df["buyExecutionMethod"].dropna().unique()).issubset({"Cash","Non cash"})
    assert set(df["sellExecutionMethod"].dropna().unique()).issubset({"Cash","Non cash"})
    assert set(df["accountOnWhoseBehalf"].dropna().unique()).issubset({"OnItsBehalf","OnCustBehalf"})
    assert set(df["nameOnWhoseBehalf"].dropna().unique()).issubset({"OnItsName","OnCustName"})
    assert set(df["transactionType"].dropna().unique()).issubset({"Direct","Indirect"})
    assert df["timePeriod"].str.match(
        r"^\d{2}:\d{2}:\d{2}\s–\s\d{2}:\d{2}:\d{2}$"
    ).all()

    def date_ge(a, b):
        return datetime.strptime(a, "%d/%m/%Y") >= datetime.strptime(b, "%d/%m/%Y")

    assert df.apply(
        lambda r: date_ge(r["transactionExecutionDate"], r["transactionSigningDate"]),
        axis=1
    ).all()

    required_cols = [
        "external_id","parent_message_id","branchId",
        "buyCurrency","sellCurrency","exchangeRate","amountRange",
        "buyExecutionMethod","sellExecutionMethod",
        "accountOnWhoseBehalf","nameOnWhoseBehalf","transactionType",
        "transactionSigningDate","SigningTime","timePeriod",
        "transactionSigningPlace","transactionExecutionEnvironment",
    ]
    for c in required_cols:
        assert df[c].notna().all(), f"{c} is required (1..1)"

    if mask_partner_mand.any():
        must_cols = [
            "partner_country","partner_residency","partner_legalStatus",
            "partner_sectOralAffiliation","partner_economicBranch",
            "partner_linkToFinancialInstitution"
        ]
        assert df.loc[mask_partner_mand, must_cols].notna().all().all(), \
            "Partner classifiers must be present when mandatory"

    assert (df.loc[mask_plegal_present & mask_nat, "partner_sectOralAffiliation"] == "Household").all()
    if df["partner_legalStatus"].notna().any():
        assert df.loc[
            ~df["partner_legalStatus"].isin(["Leg","SEnt"]), "partner_codeByCBA"
        ].isna().all()

    assert df.loc[
        mask_oncust & (df["client_legalStatus"]=="Nat"), "client_ssn"
    ].notna().all()
    assert df.loc[
        ~(mask_oncust & (df["client_legalStatus"]=="Nat")), "client_ssn"
    ].isna().all()

    mask_cpass1 = mask_oncust & (df["client_legalStatus"]=="Nat") & df["client_ssn"].isna()
    assert df.loc[mask_cpass1, "client_passport"].notna().all()

    mask_leglike_client2 = df["client_legalStatus"].isin(["Leg","SEnt","NA"])
    mask_client_resident2 = df["client_residency"] == "Resident"
    assert df.loc[
        mask_oncust & mask_leglike_client2 & mask_client_resident2, "client_tin"
    ].notna().all()
    assert df.loc[
        mask_oncust & ~mask_leglike_client2, "client_tin"
    ].isna().all()

    mask_ctin_null = df["client_tin"].isna()
    assert df.loc[
        mask_oncust & mask_leglike_client2 & mask_ctin_null, "client_legalName"
    ].notna().all()
    assert df.loc[
        mask_oncust & ~mask_leglike_client2, "client_legalName"
    ].isna().all()

    assert (
        df.loc[
            df[[
                "intermediary_tin","intermediary_legalName",
                "intermediary_codeByCBA","intermediary_linkToFinancialInstitution"
            ]].notna().any(axis=1),
            ["intermediary_country","intermediary_residency"]
        ].notna().all(axis=1)
    ).all()
    assert df.apply(env_allowed, axis=1).all()

    df = df.drop(columns=["_amdAmount_AMD","_internal_buy_vol","_internal_sell_vol"])
    return df
