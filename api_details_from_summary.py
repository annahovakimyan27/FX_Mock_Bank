# api_details_from_summary.py
# FastAPI that returns (and now also patches) the detailed rows for a given SUMMARY external_id.
#
# - GET /details/{summary_id}
#   -> {"summary_id": "...", "count": N, "details": [ {...}, ... ]}
#
# - PATCH /details/by-external-id/{detail_id}
#   -> takes a partial JSON payload of fields to change in one detailed row
#
# Primary lookup path (preferred, per regulation):
#   detailed.parent_message_id == <summary external_id>
#
# Fallback path (if parent_message_id column is missing):
#   reconstruct equality filters from the summary JSON to the detailed DF.
#
# Run:
#   uvicorn api_details_from_summary:app --reload

from fastapi import FastAPI, HTTPException, Body  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import Any, Dict, List, Optional
import os
import json
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore # add this import at the top

#

# ----------------------------
# Config — adjust paths if needed
# ----------------------------
INPUT_XLSX = "/Users/apple/Json Structure/fx_details_testtest_with_parent.xlsx"  # detailed Excel (with parent_message_id)
OUTPUT_DIR = "/Users/apple/Downloads/cba_local_sim_swagger/out_summaries7"  # where summary JSONs live

# ----------------------------
# Grouping keys (must match summarizer’s GROUP_BY) — used only for fallback
# ----------------------------
GROUP_BY = [
    "organization_id",      # per-bank separation
    "branchId",

    # Partner (7–13)
    "partner_ssn", "partner_passport", "partner_tin", "partner_legalName",
    "partner_country", "partner_residency", "partner_legalStatus",

    # Client (18–24)
    "client_ssn", "client_passport", "client_tin", "client_legalName",
    "client_country", "client_residency", "client_legalStatus",

    # Intermediary (27–32)
    "intermediary_tin", "intermediary_legalName",
    "intermediary_country", "intermediary_residency",
    "intermediary_codeByCBA", "intermediary_linkToFinancialInstitution",

    # 35–39
    "buyCurrency", "sellCurrency", "exchangeRate", "publishedExchangeRate", "amountRange",

    # 41–43
    "buyExecutionMethod", "sellExecutionMethod", "accountOnWhoseBehalf",

    # 45–46
    "nameOnWhoseBehalf", "transactionType",

    # 48–50
    "timePeriod", "transactionSigningPlace", "transactionExecutionEnvironment",
]

# Columns whose changes can affect summary aggregates but do NOT move
# the row between groups (unless GROUP_BY fields change).
AFFECTS_SUMMARY_FIELDS = {
    "buyVolume",
    "sellVolume",
    "exchangeRate",
    "publishedExchangeRate",
    "commissionFee_AMD",
}

# ----------------------------
# Map from summary JSON structure -> detailed DF column (for fallback)
# ----------------------------
SUMMARY_TO_DF = {
    # Top-level
    "organization_id": ("top", "organization_id"),
    "branchId": ("top", "branchId"),

    # Partner (7–13)
    "partner_ssn": ("partner", "ssn"),
    "partner_passport": ("partner", "passport"),
    "partner_tin": ("partner", "tin"),
    "partner_legalName": ("partner", "legalName"),
    "partner_country": ("partner", "country"),
    "partner_residency": ("partner", "residency"),
    "partner_legalStatus": ("partner", "legalStatus"),

    # Client (18–24)
    "client_ssn": ("client", "ssn"),
    "client_passport": ("client", "passport"),
    "client_tin": ("client", "tin"),
    "client_legalName": ("client", "legalName"),
    "client_country": ("client", "country"),
    "client_residency": ("client", "residency"),
    "client_legalStatus": ("client", "legalStatus"),

    # Intermediary (27–32)
    "intermediary_tin": ("intermediary", "tin"),
    "intermediary_legalName": ("intermediary", "legalName"),
    "intermediary_country": ("intermediary", "country"),
    "intermediary_residency": ("intermediary", "residency"),
    "intermediary_codeByCBA": ("intermediary", "codeByCBA"),
    "intermediary_linkToFinancialInstitution": ("intermediary", "linkToFinancialInstitution"),

    # Transaction / 35–39, 41–43, 45–46, 48–50
    "buyCurrency": ("transaction", "buyCurrency"),
    "sellCurrency": ("transaction", "sellCurrency"),
    "exchangeRate": ("transaction", "exchangeRate"),
    "publishedExchangeRate": ("transaction", "publishedExchangeRate"),
    "amountRange": ("transaction", "amountRange"),
    "buyExecutionMethod": ("transaction", "buyExecutionMethod"),
    "sellExecutionMethod": ("transaction", "sellExecutionMethod"),
    "accountOnWhoseBehalf": ("transaction", "accountOnWhoseBehalf"),
    "nameOnWhoseBehalf": ("transaction", "nameOnWhoseBehalf"),
    "transactionType": ("transaction", "transactionType"),
    "timePeriod": ("transaction", "timePeriod"),
    "transactionSigningPlace": ("transaction", "transactionSigningPlace"),
    "transactionExecutionEnvironment": ("transaction", "transactionExecutionEnvironment"),
}

# ----------------------------
# Load detailed DF (as strings for robust equality)
# ----------------------------
def _load_detailed_df() -> pd.DataFrame:
    df = pd.read_excel(INPUT_XLSX, dtype=str)
    # Normalize NaN-like values
    df = df.replace({np.nan: None})
    return df


DF_DETAILED = _load_detailed_df()

# ----------------------------
# Utilities
# ----------------------------
def _load_summary_json(summary_id: str) -> Dict[str, Any]:
    """
    Load a single summary JSON by its external_id (filename-based).
    """
    path = os.path.join(OUTPUT_DIR, f"{summary_id}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _value_from_summary(summary_obj: Dict[str, Any], key: str) -> Optional[str]:
    """
    Extract grouping value from the summary object using SUMMARY_TO_DF routing.
    Returns string or None. Used only in fallback.
    """
    where = SUMMARY_TO_DF[key]
    if where[0] == "top":
        return None if summary_obj.get(where[1]) is None else str(summary_obj.get(where[1]))
    section, field = where
    section_obj = summary_obj.get(section)
    if section_obj is None:
        return None
    val = section_obj.get(field)
    return None if val is None else str(val)


def _build_filters_from_summary(summary_obj: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Build a dict of DF column -> expected value, for all GROUP_BY columns (fallback).
    """
    filters: Dict[str, Optional[str]] = {}
    for col in GROUP_BY:
        filters[col] = _value_from_summary(summary_obj, col)
    return filters


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Apply equality/None filters to DF (fallback). DF is string-typed.
    """
    mask = pd.Series([True] * len(df))
    for col, expected in filters.items():
        if col not in df.columns:
            return df.iloc[0:0]
        if expected is None:
            mask = mask & (
                df[col].isna()
                | (df[col] == "")
                | (df[col].str.lower() == "none")
            )
        else:
            mask = mask & (df[col].fillna("").astype(str) == expected)
    return df[mask]

# ----------------------------
# Numeric / aggregation helpers (mirroring summarizer)
# ----------------------------
def _num(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return None
    try:
        s = s.astype(float)
    except Exception:
        s = pd.to_numeric(s, errors="coerce").dropna()
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


def _recompute_summary_for(summary_id: str) -> bool:
    """
    Recompute transaction statistics for a given summary_id.

    Uses DF_DETAILED rows where parent_message_id == summary_id.
    Updates or deletes the corresponding summary JSON in OUTPUT_DIR.
    """
    global DF_DETAILED

    if "parent_message_id" not in DF_DETAILED.columns:
        return False

    group = DF_DETAILED[DF_DETAILED["parent_message_id"] == summary_id]

    path = os.path.join(OUTPUT_DIR, f"{summary_id}.json")
    if not os.path.isfile(path):
        return False

    if group.empty:
        # No more rows for this summary → delete summary JSON
        try:
            os.remove(path)
        except OSError:
            pass
        return True

    with open(path, "r", encoding="utf-8") as f:
        summary_obj = json.load(f)

    txn = summary_obj.get("transaction") or {}

    def _col(df: pd.DataFrame, name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series([], dtype=float)
        return pd.to_numeric(df[name], errors="coerce")

    buy_s = _col(group, "buyVolume")
    sell_s = _col(group, "sellVolume")
    rate_s = _col(group, "exchangeRate")
    pub_rate_s = _col(group, "publishedExchangeRate")

    has_buy = not buy_s.dropna().empty
    has_sell = not sell_s.dropna().empty

    total_buy = float(round(buy_s.dropna().sum(), 2)) if has_buy else None
    total_sell = float(round(sell_s.dropna().sum(), 2)) if has_sell else None
    main_series = buy_s if has_buy else sell_s

    txn["buyVolume"] = total_buy
    txn["sellVolume"] = total_sell
    txn["totalBuyVolume"] = total_buy
    txn["totalSellVolume"] = total_sell

    txn["exchangeRate"] = _median(rate_s) if not rate_s.dropna().empty else None
    txn["publishedExchangeRate"] = _median(pub_rate_s) if not pub_rate_s.dropna().empty else None

    txn["minimumBuyVolume"] = _min(buy_s) if has_buy else None
    txn["maximumBuyVolume"] = _max(buy_s) if has_buy else None
    txn["minimumSellVolume"] = _min(sell_s) if has_sell else None
    txn["maximumSellVolume"] = _max(sell_s) if has_sell else None
    txn["median"] = _median(main_series) if main_series is not None else None
    txn["standardDeviation"] = _std(main_series) if main_series is not None else None
    txn["numberOfTransactions"] = int(len(group))

    summary_obj["transaction"] = txn

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    return True


class DetailPatchResult(BaseModel):
    detail_id: str
    summary_id: Optional[str]
    removed_from_summary: bool
    summary_recomputed: bool

class SummaryDeleteResult(BaseModel):
    summary_id: str
    cleared_rows: int
    summary_deleted: bool



def _apply_detail_patch(detail_id: str, payload: Dict[str, Any]) -> DetailPatchResult:
    """
    Apply a partial update to a single detailed row (by external_id).

    Behaviour:
    - If any GROUP_BY field changes → row is removed from its summary
      (parent_message_id set to None) and that summary is recomputed/deleted.
    - Else, if only non-grouping fields change but one of them is in
      AFFECTS_SUMMARY_FIELDS → recompute the summary statistics.
    - Otherwise, only the detailed row is updated.
    """
    global DF_DETAILED

    if "external_id" not in DF_DETAILED.columns:
        raise HTTPException(status_code=500, detail="Detailed DF has no 'external_id' column.")

    mask = DF_DETAILED["external_id"] == detail_id
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Detailed row with external_id={detail_id} not found.")

    idx = DF_DETAILED.index[mask][0]
    row_before = DF_DETAILED.loc[idx].copy()

    summary_id = row_before.get("parent_message_id")
    group_by_changed = False
    affects_summary = False

    for col, new_val in payload.items():
        if col in GROUP_BY:
            old_val = row_before.get(col)
            if (old_val is None and new_val not in (None, "")) or (
                old_val is not None and str(old_val) != str(new_val)
            ):
                group_by_changed = True
        if col in AFFECTS_SUMMARY_FIELDS:
            affects_summary = True

    # Apply the patch into DF_DETAILED
    for col, new_val in payload.items():
        if col not in DF_DETAILED.columns:
            # ignore unknown columns; alternatively, raise HTTPException
            continue
        DF_DETAILED.at[idx, col] = new_val

    removed_from_summary = False
    summary_recomputed = False

    if summary_id:
        if group_by_changed:
            DF_DETAILED.at[idx, "parent_message_id"] = None
            removed_from_summary = True
            summary_recomputed = _recompute_summary_for(summary_id)
        elif affects_summary:
            summary_recomputed = _recompute_summary_for(summary_id)

    # Persist to Excel so that future runs see the updated data
    DF_DETAILED.to_excel(INPUT_XLSX, index=False)

    return DetailPatchResult(
        detail_id=detail_id,
        summary_id=summary_id,
        removed_from_summary=removed_from_summary,
        summary_recomputed=summary_recomputed,
    )

# ----------------------------
# API Models
# ----------------------------
class DetailsResponse(BaseModel):
    summary_id: str
    count: int
    details: List[Dict[str, Any]]

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="FX Details by Summary ID", version="1.2.0")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/details/{summary_id}", response_model=DetailsResponse)
def get_details(summary_id: str):
    """
    Given a summary JSON external_id, return all detailed rows that formed it.

    Preferred path:
      - Filter DF_DETAILED where parent_message_id == summary_id
        (and, if available in summary JSON, organization_id must match too).

    Fallback path (when parent_message_id column is missing):
      - Reconstruct equality filters from the summary JSON per GROUP_BY
        and apply them to the detailed DF.
    """
    # Load the summary object (also validates existence)
    try:
        summary_obj = _load_summary_json(summary_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Summary JSON not found",
                "summary_id": summary_id,
            },
        )

    # Always load the latest detailed data from disk
    df = _load_detailed_df()

    # Primary path: parent_message_id column exists in detailed DF
    if "parent_message_id" in df.columns:
        # --- FIX: normalize strings before comparing (Excel often has trailing spaces) ---
        parent_col = df["parent_message_id"].fillna("").astype(str).str.strip()

        # Base filter: parent_message_id equals the summary external_id
        mask = (parent_col == str(summary_id).strip())

        # Tighten by organization_id when present in summary (prevents cross-bank collisions)
        org_in_summary = summary_obj.get("organization_id")
        if org_in_summary and "organization_id" in df.columns:
            org_col = df["organization_id"].fillna("").astype(str).str.strip()
            mask = mask & (org_col == str(org_in_summary).strip())

        matches = df[mask]

        if matches.empty:
            # Fall back to reconstruction (older files or not yet backfilled)
            filters = _build_filters_from_summary(summary_obj)
            matches = _apply_filters(df, filters)

            if matches.empty:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": "No detailed rows match this summary (checked parent_message_id and field-based fallback).",
                        "summary_id": summary_id,
                    },
                )
    else:
        # No parent_message_id column — use fallback only
        filters = _build_filters_from_summary(summary_obj)
        matches = _apply_filters(df, filters)
        if matches.empty:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "No detailed rows match this summary (field-based fallback).",
                    "summary_id": summary_id,
                },
            )

    # Return all column/value pairs for each row (as JSON dicts)
    details = matches.replace({np.nan: None}).to_dict(orient="records")
    return DetailsResponse(summary_id=summary_id, count=len(details), details=details)


@app.patch("/details/by-external-id/{detail_id}", response_model=DetailPatchResult)
def patch_detail(detail_id: str, payload: Dict[str, Any] = Body(...)):
    """
    Patch a single detailed row by its external_id.

    Typical flow:
    1. GET /details/{summary_id} to fetch all details and show them in an editable UI.
    2. When a user edits one row, send PATCH /details/by-external-id/{external_id}
       with only the changed fields.

    Behaviour:
    - If any GROUP_BY column changes → the row is removed from its current summary
      and that summary is recalculated (or deleted if it has no rows left).
    - If only non-grouping but summary-affecting columns change
      (e.g. buyVolume, sellVolume, rates, commissionFee_AMD) → summary aggregates
      are recalculated.
    - Otherwise → only the detailed row is updated.
    """
    return _apply_detail_patch(detail_id, payload)


@app.get("/detail/by-external-id/{detail_id}")
def get_detail_by_external_id(detail_id: str):
    """
    Return a single detailed row by its external_id.
    Useful after PATCH to inspect the updated row.
    """
    global DF_DETAILED

    if "external_id" not in DF_DETAILED.columns:
        raise HTTPException(status_code=500, detail="Detailed DF has no 'external_id' column.")

    mask = DF_DETAILED["external_id"] == detail_id
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Detail with external_id={detail_id} not found.")

    row = DF_DETAILED.loc[mask].replace({np.nan: None}).to_dict(orient="records")[0]
    return row


@app.get("/summary/{summary_id}")
def get_summary(summary_id: str):
    """
    Return the summary JSON from OUTPUT_DIR for the given summary_id.
    Used by the CBA receiver as the LIVE summary view.
    """
    path = os.path.join(OUTPUT_DIR, f"{summary_id}.json")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail={"message": "Summary JSON not found", "summary_id": summary_id},
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # just return the JSON as-is
    return JSONResponse(content=data)


@app.delete("/summary/{summary_id}", response_model=SummaryDeleteResult)
def delete_summary(summary_id: str):
    """
    Delete a summary and detach all detailed rows that were linked to it.

    Behaviour:
    - Removes the summary JSON from OUTPUT_DIR (if exists)
    - For all detailed rows with parent_message_id == summary_id:
        * sets parent_message_id = None
    - Persists the updated detailed data back to INPUT_XLSX
    """
    global DF_DETAILED

    # 1) Clear parent_message_id on all matching rows
    cleared_rows = 0
    if "parent_message_id" in DF_DETAILED.columns:
        mask = DF_DETAILED["parent_message_id"] == summary_id
        cleared_rows = int(mask.sum())
        if cleared_rows > 0:
            DF_DETAILED.loc[mask, "parent_message_id"] = None

    # 2) Delete the summary JSON file
    path = os.path.join(OUTPUT_DIR, f"{summary_id}.json")
    summary_deleted = False
    if os.path.exists(path):
        os.remove(path)
        summary_deleted = True

    # 3) If nothing existed, return 404 to signal "unknown summary"
    if cleared_rows == 0 and not summary_deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "No such summary: nothing to delete.",
                "summary_id": summary_id,
            },
        )

    # 4) Persist changes to the Excel so future runs see the reset rows
    DF_DETAILED.to_excel(INPUT_XLSX, index=False)

    return SummaryDeleteResult(
        summary_id=summary_id,
        cleared_rows=cleared_rows,
        summary_deleted=summary_deleted,
    )
