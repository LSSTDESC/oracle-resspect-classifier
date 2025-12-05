from astropy.table import Table
import numpy as np


def polars_row_to_astropy_table(row, time_dependent_features=None, prefer_numpy=True):
    """
    Convert a single Polars row (or dict-like) into an Astropy Table where each row
    corresponds to one observation (per time-step). Non-list values are stored in
    the returned Table.meta dictionary.

    Parameters
    - row: a single-row object returned from Polars (e.g. `parquet[i]`) or a mapping/dict
    - time_dependent_features: optional list of column names to treat as time-dependent lists;
      if None the function will infer list-like columns by checking for Python `list` instances
    - prefer_numpy: if True, table columns will contain numpy arrays where possible

    Returns
    - astropy.table.Table with one row per observation and metadata in `Table.meta`
    """

    # Normalize input to a plain dict
    if hasattr(row, "to_dict"):
        try:
            row_dict = row.to_dict()
        except Exception:
            # Polars sometimes returns a single-row DataFrame; try converting via dict comprehension
            row_dict = {c: row[c] for c in row.columns}
    elif isinstance(row, dict):
        row_dict = row.copy()
    else:
        # Generic fallback for mapping-like objects
        try:
            keys = getattr(row, "keys", None)
            if callable(keys):
                row_dict = {k: row[k] for k in row.keys()}
            else:
                row_dict = dict(row)
        except Exception:
            raise TypeError("Unsupported row type for conversion to dict")

    # Detect list-like columns by checking for plain Python lists (guaranteed by user)
    if time_dependent_features is None:
        time_cols = [k for k, v in row_dict.items() if isinstance(v, np.ndarray)]
    else:
        time_cols = list(time_dependent_features)

    # Build arrays for each time-dependent column and track lengths
    arrays = {}
    lengths = set()
    for col in time_cols:
        v = row_dict.get(col)
        arr = np.asarray(v)
        arr = arr.ravel()
        arrays[col] = arr
        lengths.add(arr.shape[0])

    # If lengths differ, pad numeric arrays with np.nan and object arrays with None
    if len(lengths) == 0:
        # No time-dependent columns found -> return empty table with metadata
        tbl = Table()
        tbl.meta.update({k: (v.item() if isinstance(v, np.generic) else v) for k, v in row_dict.items()})
        return tbl

    if len(lengths) > 1:
        maxlen = max(lengths)
        for col, arr in arrays.items():
            if arr.shape[0] < maxlen:
                if np.issubdtype(arr.dtype, np.number):
                    pad = np.full((maxlen - arr.shape[0],), np.nan, dtype=arr.dtype)
                else:
                    pad = np.full((maxlen - arr.shape[0],), None, dtype=object)
                arrays[col] = np.concatenate([arr, pad])

    # Create Astropy Table and populate columns
    tbl = Table()
    for col, arr in arrays.items():
        if prefer_numpy:
            tbl[col] = np.asarray(arr)
        else:
            tbl[col] = list(arr)

    # Store non-time-dependent scalar values as metadata (convert numpy scalars to Python types)
    meta = {}
    for k, v in row_dict.items():
        if k in time_cols:
            continue
        if isinstance(v, np.generic):
            meta[k] = v.item()
        else:
            meta[k] = v

    tbl.meta.update(meta)
    return tbl


# Example usage (uncomment to run):
row = data.iloc[0]         # or example_input[0]
tbl = polars_row_to_astropy_table(row)
print(tbl.meta)
print(tbl)