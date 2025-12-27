import streamlit as st
import pandas as pd
import numpy as np
import re
import pyodbc
from dateutil import parser
import sqlalchemy
import urllib

st.set_page_config(page_title="Automatic Data Cleaner", layout="wide")
st.title("🧹 Automatic Data Cleaner")

#-------------------
# Helpers
# -------------------

def robust_parse_date(val):
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    for dayfirst in (True, False):
        try:
            dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=dayfirst)
            if not pd.isna(dt):
                return dt
        except Exception:
            pass
    try:
        dt2 = parser.parse(s, fuzzy=True, dayfirst=True)
        return pd.to_datetime(dt2)
    except Exception:
        return pd.NaT

def canonical_cell(val):
    if pd.isna(val):
        return "<NA>"
    s = val
    try:
        num = float(s)
        if np.isfinite(num):
            return str(int(num)) if num.is_integer() else str(num)
    except Exception:
        pass
    try:
        dt = robust_parse_date(s)
        if not pd.isna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    out = re.sub(r"\s+", " ", str(s)).strip().lower()
    return out

def remove_duplicates_by_canonical(df, keep='first'):
    norm_df = df.applymap(canonical_cell)
    norm_keys = norm_df.apply(lambda r: "||".join(r.values.astype(str)), axis=1)
    keep_mask = ~norm_keys.duplicated(keep=keep)
    deduped = df.loc[keep_mask].copy()
    return deduped, norm_df, keep_mask

def find_and_replace(df, find_text, replace_text, column=None):
    if column and column in df.columns:
        df[column] = df[column].astype(str).str.replace(find_text, replace_text, regex=True)
    else:
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(find_text, replace_text, regex=True)
    return df

def load_sql_table(conn_str, table_name):
    conn = pyodbc.connect(conn_str)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def get_sql_tables(conn_str):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    tables = [row.table_name for row in cursor.tables(tableType='TABLE')]
    conn.close()
    return tables

def write_to_sql(df, conn_str, table_name, mode="overwrite"):
    conn_str_enc = urllib.parse.quote_plus(conn_str)
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str_enc}")
    
    if mode == "overwrite":
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    else:
        df.to_sql(table_name, con=engine, if_exists="append", index=False)

# -------------------
# UI / Main
# -------------------
data_source = st.radio("Select data source:", ["Upload file", "SQL Server"])
df = None

# Default preview variables to prevent NameError
preview_mode = False
preview_rows = 10

# --- SQL session memory ---
if "sql_df" not in st.session_state:
    st.session_state.sql_df = None
if "sql_conn_str" not in st.session_state:
    st.session_state.sql_conn_str = None
if "sql_table" not in st.session_state:
    st.session_state.sql_table = None

# --- Upload file ---
if data_source == "Upload file":
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

# --- SQL Server ---
elif data_source == "SQL Server":
    st.subheader("💾 SQL Server Connection")
    conn_str = st.text_input("Enter ODBC connection string (e.g., DRIVER={SQL Server};SERVER=localhost;DATABASE=TestDB;Trusted_Connection=yes;)")

    if conn_str:
        try:
            tables = get_sql_tables(conn_str)
            selected_table = st.selectbox("Select a table to import:", tables)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Load Table"):
                    df = load_sql_table(conn_str, selected_table)
                    st.session_state.sql_df = df
                    st.session_state.sql_conn_str = conn_str
                    st.session_state.sql_table = selected_table
                    st.success(f"Loaded {len(df)} rows from '{selected_table}'")
            with col2:
                if st.button("🧹 Clear Loaded Data"):
                    st.session_state.sql_df = None
                    st.session_state.sql_conn_str = None
                    st.session_state.sql_table = None
                    st.warning("SQL data cleared from memory.")

        except Exception as e:
            st.error(f"Error loading SQL tables: {e}")

    if st.session_state.sql_df is not None:
        df = st.session_state.sql_df
        conn_str = st.session_state.sql_conn_str
        selected_table = st.session_state.sql_table
        st.info(f"Currently loaded from: **{selected_table}** ({len(df)} rows)")

# --- Proceed if df loaded ---
if df is not None:
    st.subheader("📊 Original Data Preview")
    st.dataframe(df.head(10))

    # Cleaning options
    st.subheader("⚙️ Cleaning Options")
    remove_duplicates = st.checkbox("Remove duplicates (after normalization)", value=True)
    standardize_text = st.checkbox("Standardize text fields", value=True)
    standardize_dates = st.checkbox("Standardize date fields", value=True)

    # Preview Mode
    st.subheader("🔍 Preview Mode")
    preview_mode = st.checkbox("Enable Preview Mode", value=False)
    preview_rows = st.number_input("Number of rows to preview:", min_value=1, max_value=50, value=10)

    # Find & Replace
    st.subheader("Find & Replace")
    find_text = st.text_input("Find:")
    replace_text = st.text_input("Replace with:")
    replace_column = st.text_input("Column (leave empty for all columns):", "")

    # Column Transformations
    st.subheader("Column Transformations")
    num_transforms = st.number_input("Number of column transformations:", min_value=0, value=0, step=1)
    transformations = []
    for i in range(num_transforms):
        t_col = st.selectbox(f"Column #{i+1}:", [""] + list(df.columns), key=f"t_col_{i}")
        t_type = st.selectbox(f"Transformation type #{i+1}:", ["", "Uppercase", "Lowercase", "Title Case", "Strip Whitespace", "Remove Special Characters"], key=f"t_type_{i}")
        transformations.append((t_col, t_type))

    # Conditional Fills
    st.subheader("Conditional Fill Missing Values")
    num_cond_fills = st.number_input("Number of conditional fills:", min_value=0, value=0, step=1)
    cond_fills = []
    for i in range(num_cond_fills):
        cf_col = st.selectbox(f"Column #{i+1}:", [""] + list(df.columns), key=f"cf_col_{i}")
        cf_val = st.text_input(f"Fill value #{i+1}:", "", key=f"cf_val_{i}")
        cf_cond_col = st.selectbox(f"Conditional column (optional) #{i+1}:", [""] + list(df.columns), key=f"cf_cond_col_{i}")
        cf_cond_val = st.text_input(f"Conditional value (optional) #{i+1}:", "", key=f"cf_cond_val_{i}")
        cond_fills.append((cf_col, cf_val, cf_cond_col, cf_cond_val))

    # Missing value defaults
    st.subheader("Missing Value Defaults")
    fill_text = st.text_input("Text fill:", "Unknown")
    fill_numeric = st.text_input("Numeric fill:", "0")
    fill_date = st.text_input("Date fill:", "Unknown")

    # Run Cleaning
    if st.button("🚀 Run Cleaning"):
        st.info("Running cleaning pipeline...")
        working = df.copy()
        original_len = len(working)

        if preview_mode:
            st.warning(f"Preview Mode ON — showing first {preview_rows} rows only.")

        # Normalize
        st.info("Normalizing data fields...")
        for col in working.columns:
            series = working[col]
            if pd.api.types.is_numeric_dtype(series):
                try:
                    num_fill = float(fill_numeric)
                except:
                    num_fill = 0.0
                working[col] = series.fillna(num_fill)
            else:
                name_is_date = "date" in col.lower()
                if name_is_date and standardize_dates:
                    parsed = series.apply(lambda x: robust_parse_date(x))
                    formatted = parsed.dt.strftime("%Y-%m-%d")
                    working[col] = formatted.fillna(fill_date)
                else:
                    working[col] = series.fillna(fill_text).astype(str)
                    if standardize_text:
                        if "email" in col.lower():
                            working[col] = working[col].str.strip().str.lower().apply(lambda x: x if re.match(r"[^@]+@[^@]+\.[^@]+", x) else fill_text)
                        else:
                            working[col] = working[col].apply(lambda s: re.sub(r"\s+", " ", s).strip().title())

        # Find & Replace
        if find_text and replace_text:
            col_to_use = replace_column if replace_column.strip() else None
            working = find_and_replace(working, find_text, replace_text, column=col_to_use)
            st.success(f"Find & Replace applied: '{find_text}' → '{replace_text}'")

        # Column transformations
        for t_col, t_type in transformations:
            if t_col and t_type:
                if t_type == "Uppercase":
                    working[t_col] = working[t_col].astype(str).str.upper()
                elif t_type == "Lowercase":
                    working[t_col] = working[t_col].astype(str).str.lower()
                elif t_type == "Title Case":
                    working[t_col] = working[t_col].astype(str).str.title()
                elif t_type == "Strip Whitespace":
                    working[t_col] = working[t_col].astype(str).str.strip()
                elif t_type == "Remove Special Characters":
                    working[t_col] = working[t_col].astype(str).str.replace(r'[^A-Za-z0-9\s]', '', regex=True)

        # Conditional fills
        for cf_col, cf_val, cf_cond_col, cf_cond_val in cond_fills:
            if cf_col and cf_val:
                if cf_cond_col and cf_cond_val:
                    mask = working[cf_cond_col].astype(str) == cf_cond_val
                    working.loc[mask, cf_col] = working.loc[mask, cf_col].fillna(cf_val)
                else:
                    working[cf_col] = working[cf_col].fillna(cf_val)

        # Deduplicate
        if remove_duplicates:
            deduped_df, norm_df, keep_mask = remove_duplicates_by_canonical(working, keep='first')
            removed_count = original_len - len(deduped_df)
            working = deduped_df
            st.success(f"Duplicates removed: {removed_count}")
            st.caption("Sample normalized values:")
            st.dataframe(norm_df.head(10))

        # Restore numeric types
        for col in df.select_dtypes(include=[np.number]).columns.tolist():
            if col in working.columns:
                working[col] = pd.to_numeric(working[col], errors='coerce')

        final_len = len(working)
        st.success(f"Cleaning finished — rows before: {original_len}, after: {final_len}")

        # Display
        display_working = working.head(preview_rows) if preview_mode else working
        st.subheader("📈 Final Output")
        st.dataframe(display_working)

        # Save SQL
        if data_source == "SQL Server" and conn_str:
            st.subheader("💾 Save Cleaned Data to SQL Server")
            default_table = st.session_state.sql_table if st.session_state.sql_table else "CleanedData"
            save_table = st.text_input("Destination table name:", default_table)
            save_mode = st.selectbox("Write mode:", ["overwrite", "append"])
            confirm_save = st.checkbox(f"Confirm write to table '{save_table}'", value=False)

            if st.button("⬆️ Write to SQL"):
                if not confirm_save:
                    st.warning("Please confirm before writing to SQL.")
                else:
                    try:
                        # Backup if overwriting
                        if save_mode == "overwrite":
                            conn = pyodbc.connect(conn_str)
                            cursor = conn.cursor()
                            backup_name = f"{save_table}_Backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                            st.info(f"Backing up table '{save_table}' to '{backup_name}'...")
                            try:
                                cursor.execute(f"SELECT * INTO {backup_name} FROM {save_table}")
                                conn.commit()
                                st.success(f"Backup created: {backup_name}")
                            except Exception as e:
                                st.warning(f"Backup skipped/failure: {e}")
                            conn.close()

                        write_to_sql(working, conn_str, save_table, mode=save_mode)
                        st.success(f"Cleaned data written to '{save_table}' ({save_mode})")
                    except Exception as e:
                        st.error(f"Failed to write to SQL: {e}")

        # Download CSV
        if not preview_mode:
            csv_out = working.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download cleaned CSV", data=csv_out, file_name="cleaned_dataset.csv", mime="text/csv")
        else:
            st.info("Preview Mode active — download disabled.")
