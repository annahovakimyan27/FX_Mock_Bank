````
# FX_Mock_Bank

FX_Mock_Bank is a small toolkit for generating and summarizing **synthetic FX transaction data**.

It mimics a “local FX bank → central bank” reporting flow:

1. **Generate** detailed FX transactions for multiple banks.
2. **Group & summarize** those transactions according to regulatory rules.
3. **Link** each detailed row back to its summary via a parent ID.

---

## Repository structure

Typical layout:

```text
FX_Mock_Bank/
├── fx_data_generator2.py          # Generate detailed FX transactions (Excel)
├── sum_jsons2.py                   # Build FX summary JSONs from the details
├── fx_details_test.xlsx            # Generated detailed data (output)
├── fx_details_test_with_parent.xlsx# Detailed data with parent summary IDs (output)
└── out_summaries/                 # Summary JSON/JSONL files (output)
````

---

## 1. Detailed data generator – `fx_data_generator2.py`

This script generates a **synthetic FX “detailed” dataset** for multiple banks and applies a rich set of **data quality and business-rule checks** that mirror the FX reporting specification.

It is intended for:

* Testing **FX detailed report** ingestion and validation
* Building **end-to-end demos** of detailed → summarized reporting
* Producing **clean, rule-compliant sample data** for analysis or QA

### Key functionality

* Generates **N_ROWS** detailed FX transactions (default: 100) for **two banks**.
* Produces realistic values for:

  * Currencies, FX rates, buy/sell volumes
  * Dates & times, including a **timePeriod** bucket
  * Partner, client and intermediary attributes (country, residency, legal status, etc.)
* Enforces a wide range of **schema and business rules**, for example:

  * When identity fields (SSN, passport, TIN, legal name) are allowed, required, or forbidden
  * When client and intermediary blocks must be present or absent
  * Where classifier / sectoral fields must be populated
  * Constraints on volumes, decimals, and exchange rates
  * Valid combinations of execution method and execution environment
* Organizes rows into a small number of **stable groups** that can later be aggregated into FX summary records (group-uniform values for key columns).

### Generated data (high level)

For each row, the script creates:

* **Identifiers**

  * `external_id` (org_code + UUID)
  * `parent_message_id` (message ID based on org, date and sequence)
  * `branchId` (simple branch code)

* **Participant blocks**

  * **Partner**: country, residency, legal status, link flag, sector, economic branch, IDs (SSN / passport / TIN / legal name) as allowed by the rules
  * **Client**: present **only** when the transaction is on customer’s behalf, then populated/cleaned according to legal status and residency
  * **Intermediary**: optional; if present, ensures consistency between country/residency and identity fields

* **FX & amounts**

  * Currency pair with a bias to AMD
  * Positive exchange rate and, optionally, a published rate
  * Either buy **or** sell volume (exclusive), with:

    * Positive values
    * Max 12 integer digits
    * Up to 2 decimal places
  * An **amount range** derived from AMD equivalent

* **Execution details**

  * Execution method per leg: `"Cash"` or `"Non cash"`
  * Execution environment chosen to respect a simple matrix, e.g.:

    * Cash legs → limited to in-person / FGIAS-type environments
    * Non-cash legs → channels like web, mobile, POS, etc., depending on direct/indirect type
  * Signed and execution date/time, plus a 30-minute **timePeriod** window in `"HH:MM:SS – HH:MM:SS"` format
  * Execution/signing place examples (branches/desks)

### Grouping for future summarization

After generation, the script partitions the dataset into a fixed number of **summary groups** (e.g. 5).

Within each group, a defined set of columns (branch, partner/client/intermediary keys, FX/rates, methods, environment, etc.) is **forced to be identical**, so:

* Each group can later be collapsed into **one summary row**, and
* The original detailed rows remain realistic but **summarizable** under the regulation rules.

### Built-in validations

Before writing the output file, the script runs multiple **assertions**, including but not limited to:

* Exactly **one** of `buyVolume` / `sellVolume` is populated per row
* Volumes are > 0, properly formatted, and within length limits
* FX rates are > 0
* Enum fields (execution method, transaction type, on-whose-behalf flags, etc.) stay within allowed values
* `timePeriod` follows the required pattern
* Execution date is **not earlier** than signing date
* All mandatory (1..1) fields are non-null
* Partner/client/intermediary identity and classifier fields obey the presence/absence rules derived from the spec

If any rule fails, the script raises an assertion error instead of writing an invalid file.

### Output

By default, the script writes:

```text
fx_details_test.xlsx
```

* One row per detailed FX transaction
* All internal helper columns removed (only reporting-relevant columns included)

Console example:

```text
Saved: fx_details_test.xlsx (rows: N)
```

### How to run

```bash
python fx_data_generator2.py
```

### Configuration

At the top of the script you can adjust:

```python
N_ROWS = 100                # Number of detailed rows
RANDOM_SEED = 42            # For reproducible output
BANKS = [{"organization_id": "10001"}, {"organization_id": "20002"}]

DATE_START = datetime(2025, 7, 1)
DATE_SPAN_DAYS = 120
TIMEPERIOD_BUCKET_MINUTES = 30
```

Modify these to:

* Generate more or fewer rows
* Change the list of banks / org codes
* Shift the date range or time bucketing granularity

to match your local FX testing scenarios.

---

## 2. Summary builder – `sum_jsons2.py`

`sum_jsons2.py` takes a **detailed FX transaction file** (Excel) and produces **summary JSON objects** by grouping rows on a set of identity fields and **summing the transaction volumes** for each group.

It also **writes back** the summary ID into the detailed file, so every detailed row knows which summary it belongs to.

This script is designed to reflect the regulatory rule:

> If fields in blocks 15․2, 7–13, 18–24, 27–32, 35–39, 41–43, 45–46, 48–50 match,
> then fields 15․33–34 (volumes) are **summed**.

### What the script does

1. **Reads** a detailed FX file from a fixed path (Excel):

   ```python
   INPUT_XLSX = "/Users/apple/Json Structure/fx_details_test.xlsx"
   ```

2. **Groups** rows by the exact set of fields required by the regulation, plus `organization_id` to keep summaries per bank:

   * Branch (15․2)
   * Partner identity + classifiers (7–13)
   * Client identity + classifiers (18–24)
   * Intermediary identity + linkage (27–32)
   * Currencies, rates, amount range (35–39)
   * Execution methods + accountOnWhoseBehalf (41–43)
   * NameOnWhoseBehalf, transactionType (45–46)
   * timePeriod, signing place, execution environment (48–50)

3. For each group, it:

   * **Sums** `buyVolume` and/or `sellVolume` across the group.
   * Computes **basic statistics** (median, standard deviation, min, max).
   * Builds a **transaction snapshot** (currencies, rates, amountRange, methods, dates, environment, stats, number of transactions).
   * Builds **partner / client / intermediary** snapshots with:

     * country, residency, legalStatus
     * sectoralAffiliation, economicBranch
     * codeByCBA, linkToFinancialInstitution
     * `establishedRelationship` derived from the link flag (for partner).

4. Assigns a **new summary `external_id`** for each group:

   ```text
   <organization_id>-<uuid4>
   ```

5. **Writes outputs**:

   * One JSON file per summary, in `OUTPUT_DIR` (e.g. `out_summaries/<external_id>.json`).
   * A **JSONL** file with all summaries: `out_summaries/summaries.jsonl`.
   * A **copy of the detailed Excel file** where the `parent_message_id` column is filled with the summary `external_id` for each row:

     ```python
     OUTPUT_DETAILS_WITH_PARENT = "fx_details_test_with_parent.xlsx"
     ```

### Input & Output

#### Input

* **Excel file** with detailed rows:

  * Path configured by `INPUT_XLSX`.
  * Must contain the fields listed in `GROUP_BY` plus:

    * Volumes: `buyVolume`, `sellVolume`
    * FX: `exchangeRate`, `publishedExchangeRate`
    * Dates: `transactionSigningDate`, `transactionExecutionDate`
    * Party classifiers and linkage fields (partner/client/intermediary)

This script is designed to work naturally with the output of `fx_data_generator2.py`.

#### Output

1. **Summary JSON files** (one per group):

   * Directory: `OUTPUT_DIR` (default: `out_summaries`)
   * File name: `<external_id>.json`
   * Structure (simplified):

     ```json
     {
       "external_id": "...",
       "organization_id": "10001",
       "branchId": "001",
       "partner": { "...": "..." },
       "client": { "...": "..." },
       "intermediary": { "...": "..." },
       "transaction": {
         "buyCurrency": "USD",
         "sellCurrency": "AMD",
         "buyVolume": 123456.78,
         "sellVolume": null,
         "exchangeRate": 400.0,
         "amountRange": "100000 – 400000",
         "timePeriod": "10:00:00 – 10:30:00",
         "numberOfTransactions": 5,
         "median": 25000.0,
         "standardDeviation": 5000.0
       }
     }
     ```

2. **All summaries in one JSONL file**:

   * `out_summaries/summaries.jsonl`
   * Each line = one JSON summary object.

3. **Updated detailed Excel file**:

   * `fx_details_test_with_parent.xlsx`
   * Same rows as input, but with `parent_message_id` populated per group:

     * Every detailed row’s `parent_message_id` = the summary `external_id` of its group.

### Grouping logic (high level)

Rows belong to the same summary group if all of the following match:

* `organization_id` and `branchId`
* Partner identity & classification fields (SSN/passport/TIN/name, country, residency, legal status)
* Client identity & classification fields (if present)
* Intermediary identity and linkage (if present)
* Currencies, exchange rates and amountRange
* Execution methods, whose behalf, transactionType
* timePeriod, signing place, execution environment

**Within each group:**

* `buyVolume` and/or `sellVolume` are **summed**
* Rates are summarized by **median**
* Amount range and environment-like fields are taken only if they are **uniform** inside the group

If fields are not consistent where they should be, the helper `_only_if_all_equal` returns `None`, which will appear as `null` in the JSON.

### How to run

From the repository root:

```bash
python sum_jsons2.py
```

The script uses fixed paths defined at the top:

```python
INPUT_XLSX = "/Users/apple/Json Structure/fx_details_test.xlsx"
OUTPUT_DIR = "out_summaries"
OUTPUT_DETAILS_WITH_PARENT = "fx_details_test_with_parent.xlsx"
```

Adjust these to your environment (or make them CLI arguments later) if needed.

Example console output:

```text
✅ Summaries written to out_summaries
✅ Detailed rows updated with parent_message_id → fx_details_test_with_parent.xlsx
```

### Customization

You can adapt the script by:

* Changing `INPUT_XLSX`, `OUTPUT_DIR`, `OUTPUT_DETAILS_WITH_PARENT`.
* Modifying the `GROUP_BY` list if the regulatory grouping keys change.
* Extending the summary payload (e.g. adding more statistics or flags) where the `transaction`, `partner`, `client`, or `intermediary` objects are built.

This keeps the **regulatory grouping rule** intact while making the format fit your internal systems.

---

## End-to-end workflow

1. Run the **generator** to produce detailed FX transactions:

   ```bash
   python fx_data_generator2.py
   ```

2. Run the **summary builder** to create summary JSONs and back-link them into the details:

   ```bash
   python sum_jsons2.py
   ```

3. Use:

   * `fx_details_test_with_parent.xlsx` for testing detailed-level ingestion with parent references.
   * JSONs in `out_summaries/` as **regulatory-like FX summaries** for integration tests, demos, or validation prototyping.

```
```
