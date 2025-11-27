# sum_jsons.py
# Fixed paths — no CLI args.
# Reads fx_details_1000rowsagain.xlsx and writes summaries into out_summaries5/
# Complies with the rule:
# «If fields in 15․2, 7–13, 18–24, 27–32, 35–39, 41–43, 45–46, 48–50 match,
#  then fields 15․33–34 are SUMMED.»
#
# Outputs match your example structure more closely:
# - transaction.buyVolume / sellVolume hold the group SUMS
# - also includes totalBuyVolume / totalSellVolume
# - partner/client/intermediary include sectoralAffiliation, economicBranch, codeByCBA, linkToFinancialInstitution
# - emits establishedRelationship derived from link flag
#
# NEW:
# - Grouping now includes organization_id (per-bank summaries).
# - After each summary is created, its external_id is written back into
#   the original detailed rows' `parent_message_id` for that group.
# - The updated details file is saved alongside the summaries.

import os
import json
import uuid
import random
import pandas as pd  # type: ignore
import numpy as np   # type: ignore

# ----------------------------
# Configuration
# ----------------------------
INPUT_XLSX = "/Users/apple/Json Structure/fx_details_test.xlsx"
OUTPUT_DIR = "out_summaries7"
OUTPUT_DETAILS_WITH_PARENT = "fx_details_test_with_parent.xlsx"

RANDOM_SEED = 123
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------
# Grouping keys (identity fields for summary objects)
# EXACTLY the fields mandated by regulation 15․2, 7–13, 18–24, 27–32, 35–39, 41–43, 45–46, 48–50
# Plus: organization_id to ensure per-bank separation.
# ----------------------------
GROUP_BY = [
    "organization_id",   # <— ensure per-bank summaries

    # 15․2 — Branch
    "branchId",

    # 15․7–13 — Partner identity + classifiers
    "partner_ssn", "partner_passport", "partner_tin", "partner_legalName",
    "partner_country", "partner_residency", "partner_legalStatus",

    # 15․18–24 — Client identity + classifiers
    "client_ssn", "client_passport", "client_tin", "client_legalName",
    "client_country", "client_residency", "client_legalStatus",

    # 15․27–32 — Intermediary identity + linkage
    "intermediary_tin", "intermediary_legalName",
    "intermediary_country", "intermediary_residency",
    "intermediary_codeByCBA", "intermediary_linkToFinancialInstitution",

    # 15․35–39 — Currencies, rates, amount range
    "buyCurrency", "sellCurrency", "exchangeRate", "publishedExchangeRate", "amountRange",

    # 15․41–43 — Execution methods + accountOnWhoseBehalf
    "buyExecutionMethod", "sellExecutionMethod", "accountOnWhoseBehalf",

    # 15․45–46 — nameOnWhoseBehalf, transactionType
    "nameOnWhoseBehalf", "transactionType",

    # 15․48–50 — timePeriod, signing place, execution environment
    "timePeriod", "transactionSigningPlace", "transactionExecutionEnvironment",
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

        txn = {
            "buyCurrency": key["buyCurrency"],
            "sellCurrency": key["sellCurrency"],

            # Regulatory sums
            "buyVolume": total_buy,
            "sellVolume": total_sell,

            # Explicit totals (duplicative but handy)
            # "totalBuyVolume": total_buy,
            # "totalSellVolume": total_sell,

            "exchangeRate": _median(g["exchangeRate"]) if "exchangeRate" in g else None,
            "publishedExchangeRate": _median(g["publishedExchangeRate"]) if "publishedExchangeRate" in g else None,

            "amountRange": _only_if_all_equal(g["amountRange"]) if "amountRange" in g else None,
            "buyExecutionMethod": key["buyExecutionMethod"],
            "sellExecutionMethod": key["sellExecutionMethod"],
            "accountOnWhoseBehalf": key["accountOnWhoseBehalf"],
            "nameOnWhoseBehalf": key["nameOnWhoseBehalf"],
            "transactionType": key["transactionType"],

            "transactionSigningDate": signing_date,
            "transactionExecutionDate": exec_date or signing_date,

            "timePeriod": key["timePeriod"],
            "transactionSigningPlace": key["transactionSigningPlace"],
            "transactionExecutionEnvironment": key["transactionExecutionEnvironment"],

            # Extra stats
            "minimumBuyVolume": _min(buy_s) if has_buy else None,
            "maximumBuyVolume": _max(buy_s) if has_buy else None,
            "minimumSellVolume": _min(sell_s) if has_sell else None,
            "maximumSellVolume": _max(sell_s) if has_sell else None,
            "median": _median(main_series),
            "standardDeviation": _std(main_series),
            "numberOfTransactions": int(len(g)),
        }

        # ----- Party snapshots -----
        partner_link = _only_if_all_equal(g["partner_linkToFinancialInstitution"]) if "partner_linkToFinancialInstitution" in g else None
        partner = {
            "establishedRelationship": _established_from_link(partner_link),
            "country": _only_if_all_equal(g["partner_country"]) if "partner_country" in g else None,
            "residency": _only_if_all_equal(g["partner_residency"]) if "partner_residency" in g else None,
            "legalStatus": _only_if_all_equal(g["partner_legalStatus"]) if "partner_legalStatus" in g else None,
            "sectoralAffiliation": _only_if_all_equal(g["partner_sectOralAffiliation"]) if "partner_sectOralAffiliation" in g else None,
            "economicBranch": _only_if_all_equal(g["partner_economicBranch"]) if "partner_economicBranch" in g else None,
            "codeByCBA": _only_if_all_equal(g["partner_codeByCBA"]) if "partner_codeByCBA" in g else None,
            "linkToFinancialInstitution": partner_link,
        }

        client = None
        if key["accountOnWhoseBehalf"] == "OnCustBehalf":
            client_link = _only_if_all_equal(g["client_linkToFinancialInstitution"]) if "client_linkToFinancialInstitution" in g else None
            client = {
                "country": _only_if_all_equal(g["client_country"]) if "client_country" in g else None,
                "residency": _only_if_all_equal(g["client_residency"]) if "client_residency" in g else None,
                "legalStatus": _only_if_all_equal(g["client_legalStatus"]) if "client_legalStatus" in g else None,
                "sectoralAffiliation": _only_if_all_equal(g["client_sectoralAffiliation"]) if "client_sectoralAffiliation" in g else None,
                "economicBranch": _only_if_all_equal(g["client_economicBranch"]) if "client_economicBranch" in g else None,
                "codeByCBA": _only_if_all_equal(g["client_codeByCBA"]) if "client_codeByCBA" in g else None,
                "linkToFinancialInstitution": client_link,
            }

        intermediary = None
        if "intermediary_country" in g and "intermediary_residency" in g:
            if g[["intermediary_country", "intermediary_residency"]].notna().any().any():
                inter_link = _only_if_all_equal(g["intermediary_linkToFinancialInstitution"]) if "intermediary_linkToFinancialInstitution" in g else None
                intermediary = {
                    "country": _only_if_all_equal(g["intermediary_country"]),
                    "residency": _only_if_all_equal(g["intermediary_residency"]),
                    "codeByCBA": _only_if_all_equal(g["intermediary_codeByCBA"]) if "intermediary_codeByCBA" in g else None,
                    "linkToFinancialInstitution": inter_link,
                }

        # Create a new summary external_id (per bank)
        external_id = f"{org}-{uuid.uuid4()}"

        # Build the summary object
        obj = {
            "external_id": external_id,
            "organization_id": org,        # optional but helpful
            "branchId": key["branchId"],
            "partner": partner,
            "client": client,
            "intermediary": intermediary,
            "transaction": txn,
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
