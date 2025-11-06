"""

"""
### MODULES #####################################
import re
import pandas as pd
import json

### FUNCTIONS ###################################
def unescape_fragment(s: str) -> str:
    """
    Convert literal escape sequences like \\n and \\\" into real newlines/quotes,
    so the fragment becomes valid JSON-like text.
    """
    try:
        return bytes(s, "utf-8").decode("unicode_escape")
    except Exception:
        # Fallback: replace common escapes
        return s.replace(r'\n', '\n').replace(r'\"', '"')

def extract_brackets(line: str, pattern: str) -> str:
    """
    Find the first {...} block that appears after `pattern` in `line`.
    Returns the block including the outer braces ('{...}'), or None if not found.
    Handles nested braces by counting.
    Works on escaped input (unescapes internally).
    """
    text = unescape_fragment(line)
    # find pattern
    pos = text.find(pattern)
    if pos == -1:
        return None
    # find first '{' after pattern
    start = text.find('{', pos)
    if start == -1:
        return None
    # walk to find matching closing brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                # include both braces
                return text[start:i+1]
    return None  # unmatched

def extract_ID(line: str) -> str:
    """
    Extract trial_id using a regex from the *unescaped* line.
    Returns the id string or None.
    """
    text = unescape_fragment(line)
    m = re.search(r'"trial_id"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else None

def extract_column_names_and_values(fragment: str):
    """
    Given a fragment string that is either:
      - a JSON object string including outer braces -> parse it with json.loads
      - or a JSON-like fragment without outer braces -> try to wrap and parse
    Returns (values_list, names_list). Values are mostly left as Python types.
    If parsing fails, falls back to a regex extraction.
    """
    if fragment is None:
        return [], []

    text = fragment.strip()
    # ensure it's a JSON object string
    if not text.startswith('{'):
        text = '{' + text + '}'

    # attempt to parse as JSON
    try:
        d = json.loads(text)
        names = list(d.keys())
        values = list(d.values())
        return values, names
    except Exception:
        # fallback: simple regex for key: value pairs (best-effort)
        pairs = re.findall(r'"([^"]+)"\s*:\s*([^,}\n]+)', text)
        names = []
        values = []
        for key, val in pairs:
            names.append(key)
            v = val.strip().strip('"')
            # try numeric conversion / booleans
            if v.lower() in ("true", "false"):
                values.append(v.lower() == "true")
            else:
                try:
                    if '.' in v:
                        values.append(float(v))
                    else:
                        values.append(int(v))
                except Exception:
                    values.append(v)
        return values, names

### MAIN EXECUTION ##############################

# 1. Load the result json as a dictionary
# Replace "results.json" with your actual filename
with open("/home/eduamgo/eduamgo/Architecture_Search/NaroNet_Search_2025-10-20_14-36-38/experiment_state-2025-10-20_14-36-39.json", "r", encoding="utf-8") as f:
    input_dict = json.load(f)

# 2. Extract the value of the key "checkpoints" as a list
checkpoint_list = input_dict.get("checkpoints", [])

# 3. Clean each element of the list to keep only the config, last result and trial ID
config_list = []
last_result_list = []
id_list = []

for entry in checkpoint_list:
    # entry might already be a dict or a string; convert to string for safe processing
    entry_str = json.dumps(entry) if not isinstance(entry, str) else entry
    config_blk = extract_brackets(entry_str, "config")
    last_result_blk = extract_brackets(entry_str, "last_result")
    tid = extract_ID(entry_str)

    config_list.append(config_blk)
    last_result_list.append(last_result_blk)
    id_list.append(tid)

print(config_blk)
print(last_result_blk)

# 4. Parse the fragments and create row dictionaries
config_rows = []
last_result_rows = []

for cfg_fragment, tid in zip(config_list, id_list):
    values, names = extract_column_names_and_values(cfg_fragment)
    row = {k: v for k, v in zip(names, values)}
    row["trial_id"] = tid
    config_rows.append(row)

for res_fragment, tid in zip(last_result_list, id_list):
    values, names = extract_column_names_and_values(res_fragment)
    row = {k: v for k, v in zip(names, values)}
    row["trial_id"] = tid
    last_result_rows.append(row)

# 5. Build DataFrames
config_df = pd.DataFrame(config_rows)
last_result_df = pd.DataFrame(last_result_rows)

# Example: show heads
print("CONFIG DF")
print(config_df.head())
print("\nLAST RESULT DF")
print(last_result_df.head())

# 6. Add acc_test column from the result dataframe to the config dataframe
config_df = config_df.merge(last_result_df[["trial_id", "acc_test"]], on="trial_id", how="left")

# 7. Output
config_df.to_csv("/home/eduamgo/eduamgo/Architecture_Search/Results/trial_configs_def.csv")
config_df.to_excel("/home/eduamgo/eduamgo/Architecture_Search/Results/trial_configs_def.xlsx", engine='openpyxl')
last_result_df.to_csv("/home/eduamgo/eduamgo/Architecture_Search/Results/trial_out_params_def.csv")
last_result_df.to_excel("/home/eduamgo/eduamgo/Architecture_Search/Results/trial_out_params_def.xlsx", engine='openpyxl')
