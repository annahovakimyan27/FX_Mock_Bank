# sum_jsons.py
# Fixed paths — no CLI args.
# Reads fx_details_test.xlsx and writes summaries into out_summaries7/
# Complies with the rule:
# «If fields in 15․2, 7–13, 18–24, 27–32, 35–39, 41–43, 45–46, 48–50 match,
#  then fields 15․33–34 are SUMMED.»
#
# Outputs:
# - FLAT one-level JSON (no nested partner/client/intermediary/transaction objects)
# - transaction_* keys hold transaction fields (including summed buyVolume/sellVolume)
# - partner_*, client_*, intermediary_* keys hold party snapshots
#
# NEW:
# - Grouping includes organization_id (per-bank summaries).
# - After each summary is created, its external_id is written back into
#   the original detailed rows' `parent_message_id` for that group.
# - The updated details file is saved alongside the summaries.

import os
import json
import uuid
import random
import re
import pandas as pd  # type: ignore
import numpy as np   # type: ignore

# ----------------------------
# Configuration
# ----------------------------
INPUT_XLSX = "/Users/apple/Json Structure/fx_details_testtest.xlsx"
OUTPUT_DIR = "out_summaries7"
OUTPUT_DETAILS_WITH_PARENT = "fx_details_testtest_with_parent.xlsx"

RANDOM_SEED = 123
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------
# Grouping keys (UPDATED grouping: new COLS_* lock sets)
# Plus: organization_id to ensure per-bank separation.
# ----------------------------
GROUP_BY = [
    "organization_id",   # <— ensure per-bank summaries

    # COLS_BRANCH
    "branchId",

    # COLS_PARTNER
    "partner_country",
    "partner_estalishedRbelatiօnship",
    "partner_legalStatus",
    "partner_sectoralAffiliation",
    "partner_economicBranch",
    "partner_codeByCBA",
    "partner_linkToFinancialInstitution",

    # COLS_CLIENT
    "client_country",
    "client_legalStatus",
    "client_sectoralAffiliation",
    "client_economicBranch",
    "client_codeByCBA",
    "client_linkToFinancialInstitution",

    # COLS_INTERMEDIARY
    "intermediary_country",
    "intermediary_codeByCBA",
    "intermediary_linkToFinancialInstitution",

    # COLS_RATES
    "buyCurrency",
    "sellCurrency",
    "exchangeRate",
    "publishedExchangeRate",
    "amountRange",

    # COLS_METHODS
    "buyExecutionMethod",
    "sellExecutionMethod",
    "accountOnWhoseBehalf",
    "nameOnWhoseBehalf",
    "transactionType",

    # COLS_DATES_TIMES
    "transactionSigningDate",
    "transactionExecutionDate",
    "timePeriod",
    "ExecutionTime",

    # COLS_ENV
    "transactionSigningPlace",
    "transactionExecutionEnvironment",
]

# ----------------------------
# Helpers
# ----------------------------
def _only_if_all_equal(series: pd.Series):
    vals = series.dropna().unique()
    return vals[0] if len(vals) == 1 else None

def _num(series: pd.Series):
    s = series.dropna().astype(float)
    return s if not s.empty else None

def _median(series: pd.Series):
    s = _num(series)
    return float(round(np.median(s), 2)) if s is not None else None

def _std(series: pd.Series):
    s = _num(series)
    return float(round(np.std(s, ddof=0), 2)) if s is not None else None

def _min(series: pd.Series):
    s = _num(series)
    return float(round(s.min(), 2)) if s is not None else None

def _max(series: pd.Series):
    s = _num(series)
    return float(round(s.max(), 2)) if s is not None else None

def _established_from_link(link_val: str):
    # Map detailed flag to summary enum
    return "EstablishedRelationship" if link_val == "Linked" else (
        "NotEstablished" if link_val is not None else None
    )

# ----------------------------
# ONLY NEW: metadata code mappings for amountRange + timePeriod
# ----------------------------
def _fmt_millions(x_m: float) -> str:
    # 0.1, 1.5 keep decimals; 20.0 -> "20"
    if abs(x_m - round(x_m)) < 1e-12:
        return str(int(round(x_m)))
    return f"{x_m:.12g}"

def _map_amount_range_to_code(val):
    """
    Excel amountRange is like: "40000000 – 80000000"
    JSON should be: "40-80"  (millions)
    Special case: "0 – 100000" -> "0.1"
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    nums = re.findall(r"\d+", s)
    if len(nums) < 2:
        return s  # fallback

    lo = int(nums[0])
    hi = int(nums[1])

    if lo == 0 and hi == 100_000:
        return "0.1"

    lo_m = lo / 1_000_000.0
    hi_m = hi / 1_000_000.0
    return f"{_fmt_millions(lo_m)}-{_fmt_millions(hi_m)}"

def _map_time_period_to_code(val):
    """
    Excel timePeriod is like: "23:30:00 – 00:00:00"
    JSON should be: "2330" (HHMM of start)
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    start = s.split("–")[0].strip()  # "HH:MM:SS"
    m = re.match(r"^(?P<h>\d{2}):(?P<m>\d{2}):\d{2}$", start)
    if not m:
        return s  # fallback
    return f"{m.group('h')}{m.group('m')}"

# ----------------------------
# Summarization
# ----------------------------
def summarize_file():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_excel(INPUT_XLSX, dtype=str)

    # Numeric columns
    for col in ["buyVolume", "sellVolume", "exchangeRate", "publishedExchangeRate", "commissionFee_AMD"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    groups = df.groupby(GROUP_BY, dropna=False)

    out_jsonl_path = os.path.join(OUTPUT_DIR, "summaries.jsonl")
    out_jsonl = open(out_jsonl_path, "w", encoding="utf-8")

    # Ensure parent_message_id exists; we'll fill it per group
    if "parent_message_id" not in df.columns:
        df["parent_message_id"] = pd.NA

    for keys, g in groups:
        key = dict(zip(GROUP_BY, keys))
        org = key["organization_id"]

        # Representative dates (not grouping keys)
        signing_date = _only_if_all_equal(g["transactionSigningDate"]) if "transactionSigningDate" in g.columns else None
        exec_date = _only_if_all_equal(g["transactionExecutionDate"]) if "transactionExecutionDate" in g.columns else None

        # Volumes (15․33–34): sum within the group
        buy_s = g["buyVolume"] if "buyVolume" in g else pd.Series([], dtype=float)
        sell_s = g["sellVolume"] if "sellVolume" in g else pd.Series([], dtype=float)
        has_buy = buy_s.notna().any()
        has_sell = sell_s.notna().any()

        total_buy = float(round(buy_s.dropna().sum(), 2)) if has_buy else None
        total_sell = float(round(sell_s.dropna().sum(), 2)) if has_sell else None
        main_series = buy_s if has_buy else sell_s

        # ----- Party snapshots (same logic, just not nested in output) -----
        partner_link = _only_if_all_equal(g["partner_linkToFinancialInstitution"]) if "partner_linkToFinancialInstitution" in g else None

        client_link = None
        if key["accountOnWhoseBehalf"] == "OnCustBehalf":
            client_link = _only_if_all_equal(g["client_linkToFinancialInstitution"]) if "client_linkToFinancialInstitution" in g else None

        inter_link = None
        intermediary_present = False
        if "intermediary_country" in g and "intermediary_residency" in g:
            intermediary_present = g[["intermediary_country", "intermediary_residency"]].notna().any().any()
            if intermediary_present:
                inter_link = _only_if_all_equal(g["intermediary_linkToFinancialInstitution"]) if "intermediary_linkToFinancialInstitution" in g else None

        # Create a new summary external_id (per bank)
        external_id = f"{org}-{uuid.uuid4()}"

        obj = {
            "external_id": external_id,
            "organization_id": org,
            "branchId": key["branchId"],

            # ---- Partner (flat) ----
            "partner_establishedRelationship": _established_from_link(partner_link),
            "partner_country": _only_if_all_equal(g["partner_country"]) if "partner_country" in g else None,
            "partner_residency": _only_if_all_equal(g["partner_residency"]) if "partner_residency" in g else None,
            "partner_legalStatus": _only_if_all_equal(g["partner_legalStatus"]) if "partner_legalStatus" in g else None,
            "partner_sectoralAffiliation": _only_if_all_equal(g["partner_sectoralAffiliation"]) if "partner_sectoralAffiliation" in g else None,
            "partner_economicBranch": _only_if_all_equal(g["partner_economicBranch"]) if "partner_economicBranch" in g else None,
            "partner_codeByCBA": _only_if_all_equal(g["partner_codeByCBA"]) if "partner_codeByCBA" in g else None,
            "partner_linkToFinancialInstitution": partner_link,

            # ---- Client (flat; keep None when not OnCustBehalf) ----
            "client_country": (_only_if_all_equal(g["client_country"]) if "client_country" in g else None) if key["accountOnWhoseBehalf"] == "OnCustBehalf" else None,
            "client_residency": (_only_if_all_equal(g["client_residency"]) if "client_residency" in g else None) if key["accountOnWhoseBehalf"] == "OnCustBehalf" else None,
            "client_legalStatus": (_only_if_all_equal(g["client_legalStatus"]) if "client_legalStatus" in g else None) if key["accountOnWhoseBehalf"] == "OnCustBehalf" else None,
            "client_sectoralAffiliation": (_only_if_all_equal(g["client_sectoralAffiliation"]) if "client_sectoralAffiliation" in g else None) if key["accountOnWhoseBehalf"] == "OnCustBehalf" else None,
            "client_economicBranch": (_only_if_all_equal(g["client_economicBranch"]) if "client_economicBranch" in g else None) if key["accountOnWhoseBehalf"] == "OnCustBehalf" else None,
            "client_codeByCBA": (_only_if_all_equal(g["client_codeByCBA"]) if "client_codeByCBA" in g else None) if key["accountOnWhoseBehalf"] == "OnCustBehalf" else None,
            "client_linkToFinancialInstitution": client_link if key["accountOnWhoseBehalf"] == "OnCustBehalf" else None,

            # ---- Intermediary (flat; keep None when not present) ----
            "intermediary_country": _only_if_all_equal(g["intermediary_country"]) if intermediary_present and "intermediary_country" in g else None,
            "intermediary_residency": _only_if_all_equal(g["intermediary_residency"]) if intermediary_present and "intermediary_residency" in g else None,
            "intermediary_codeByCBA": _only_if_all_equal(g["intermediary_codeByCBA"]) if intermediary_present and "intermediary_codeByCBA" in g else None,
            "intermediary_linkToFinancialInstitution": inter_link if intermediary_present else None,

            # ---- Transaction (flat, prefixed) ----
            "transaction_buyCurrency": key["buyCurrency"],
            "transaction_sellCurrency": key["sellCurrency"],

            "transaction_buyVolume": total_buy,
            "transaction_sellVolume": total_sell,

            "transaction_exchangeRate": _median(g["exchangeRate"]) if "exchangeRate" in g else None,
            "transaction_publishedExchangeRate": _median(g["publishedExchangeRate"]) if "publishedExchangeRate" in g else None,

            # CHANGED: write metadata code instead of label
            "transaction_amountRange": _map_amount_range_to_code(
                _only_if_all_equal(g["amountRange"]) if "amountRange" in g else None
            ),

            "transaction_buyExecutionMethod": key["buyExecutionMethod"],
            "transaction_sellExecutionMethod": key["sellExecutionMethod"],
            "transaction_accountOnWhoseBehalf": key["accountOnWhoseBehalf"],
            "transaction_nameOnWhoseBehalf": key["nameOnWhoseBehalf"],
            "transaction_transactionType": key["transactionType"],

            "transaction_transactionSigningDate": signing_date,
            "transaction_transactionExecutionDate": exec_date or signing_date,

            # CHANGED: write metadata code instead of label
            "transaction_timePeriod": _map_time_period_to_code(key["timePeriod"]),

            "transaction_transactionSigningPlace": key["transactionSigningPlace"],
            "transaction_transactionExecutionEnvironment": key["transactionExecutionEnvironment"],

            "transaction_minimumBuyVolume": _min(buy_s) if has_buy else None,
            "transaction_maximumBuyVolume": _max(buy_s) if has_buy else None,
            "transaction_minimumSellVolume": _min(sell_s) if has_sell else None,
            "transaction_maximumSellVolume": _max(sell_s) if has_sell else None,
            "transaction_median": _median(main_series),
            "transaction_standardDeviation": _std(main_series),
            "transaction_numberOfTransactions": int(len(g)),
        }

        # Write per-object JSON and append to JSONL
        path = os.path.join(OUTPUT_DIR, f"{external_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        out_jsonl.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Fill the detailed rows' parent_message_id with this summary external_id
        df.loc[g.index, "parent_message_id"] = external_id

    out_jsonl.close()
    print(f"✅ Summaries written to {OUTPUT_DIR}")

    # Save the updated detailed file with parent_message_id populated
    df.to_excel(OUTPUT_DETAILS_WITH_PARENT, index=False)
    print(f"✅ Detailed rows updated with parent_message_id → {OUTPUT_DETAILS_WITH_PARENT}")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    summarize_file()
