import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import gc
import warnings
import matplotlib.pyplot as plt
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# ============== PAGE CONFIG (ADD THIS AT THE TOP) ==============
st.set_page_config(
    page_title="PT. XYZ Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CONSTANTS ==============
MAX_ROWS_IN_MEMORY = None  # Max rows to keep in session state
MAX_ROWS_PER_PAGE = 5000     # Max rows per dataframe page
MAX_PLOT_POINTS = 2000       # Max points for plotting

# ------------------ DATABASE CONNECTION ------------------
# def get_connection():
#     """
#     Connect to Supabase PostgreSQL database
#     Credentials untuk project: db.jzrghramanqolxgzdpkf
#     """
#     try:
#         connection = psycopg2.connect(
#             host="db.jzrghramanqolxgzdpkf.supabase.co",
#             port=6543,
#             database="postgres",
#             user="postgres",
#             password="OAB6It7fUDIwYH2w",
#             cursor_factory=RealDictCursor,
#             sslmode='require'
#         )
#         connection.autocommit = False  # Manual commit
#         return connection
#     except Exception as e:
#         st.error(f"❌ Koneksi database gagal: {e}")
#         st.info("💡 Periksa credentials Supabase Anda")
#         raise


def get_connection():
    try:
        return psycopg2.connect(
            host=st.secrets["supabase"]["host"],       
            port=st.secrets["supabase"]["port"],       
            database=st.secrets["supabase"]["database"],
            user=st.secrets["supabase"]["user"],
            password=st.secrets["supabase"]["password"],
            cursor_factory=RealDictCursor,
            sslmode="require"
        )
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        raise


# ------------------ SESSION STATE INIT ------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'forecast_result' not in st.session_state:
    st.session_state.forecast_result = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'cleaning_done' not in st.session_state:
    st.session_state.cleaning_done = False
if 'db_after_clean' not in st.session_state:
    st.session_state.db_after_clean = 0
if 'admin_view_mode' not in st.session_state:
    st.session_state.admin_view_mode = "User Mode"


# ============== HELPER FUNCTIONS FOR FIXES ==============
def safe_metric_value(value):
    """Safely format value for st.metric - FIX for TypeError"""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value
    return str(value)

def check_data_size_warning(df):
    """Check if dataframe is too large - FIX for MessageSizeError"""
    if df is None:
        return True
    
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    row_count = len(df)
    
    if memory_mb > 150:
        st.warning(f"⚠️ Large dataset detected: {row_count:,} rows ({memory_mb:.2f} MB)")
        st.info("💡 Using pagination for better performance")
    
    if memory_mb > 500:
        st.error(f"❌ Dataset too large ({memory_mb:.2f} MB). Please use database queries with LIMIT.")
        return False
    
    return True

def optimize_dataframe_memory(df):
    """Optimize dataframe memory usage - FIXED to pmain(reserve datetime columns"""
    df_optimized = df.copy()
    
    # ✅ FIX: Identify datetime columns BEFORE optimization
    datetime_cols = []
    for col in df_optimized.columns:
        if pd.api.types.is_datetime64_any_dtype(df_optimized[col]):
            datetime_cols.append(col)
        elif 'date' in col.lower():
            # Try to convert if column name suggests it's a date
            try:
                df_optimized[col] = pd.to_datetime(df_optimized[col], errors='coerce')
                datetime_cols.append(col)
            except:
                pass
    
    # Optimize numeric columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        if col not in datetime_cols:  # ✅ Skip datetime columns
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
    
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        if col not in datetime_cols:  # ✅ Skip datetime columns
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Convert object columns to category if beneficial
    for col in df_optimized.select_dtypes(include=['object']).columns:
        if col not in datetime_cols:  # ✅ Skip datetime columns
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            if num_unique / num_total < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
    
    # ✅ IMPORTANT: Ensure datetime columns remain as datetime64
    for col in datetime_cols:
        if col in df_optimized.columns:
            df_optimized[col] = pd.to_datetime(df_optimized[col], errors='coerce')
    
    return df_optimized

# ------------------ DATABASE SCHEMA CHECKER ------------------
def check_and_update_schema():
    """
    Check and update database schema untuk PostgreSQL
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Check transactions table
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'transactions' 
                AND column_name = 'user_id'
            """)
            if not cursor.fetchone():
                cursor.execute("""
                    ALTER TABLE transactions 
                    ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0
                """)
                cursor.execute("""
                    CREATE INDEX idx_transactions_user_id 
                    ON transactions(user_id)
                """)
                st.info("✅ Added user_id column to transactions table")
            
            # Check forecasts table
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'forecasts' 
                AND column_name = 'user_id'
            """)
            if not cursor.fetchone():
                cursor.execute("""
                    ALTER TABLE forecasts 
                    ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0
                """)
                cursor.execute("""
                    CREATE INDEX idx_forecasts_user_id 
                    ON forecasts(user_id)
                """)
                st.info("✅ Added user_id column to forecasts table")
            
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.warning(f"Schema check: {e}")
    finally:
        conn.close()

# ------------------ Utilities ------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def save_transactions_to_db(df: pd.DataFrame, user_id: int, batch_size=1000):
    """
    Save transactions to PostgreSQL database with batch processing
    """
    if df is None or len(df) == 0:
        st.error("❌ Data kosong")
        return 0
    
    if user_id is None or user_id <= 0:
        st.error("❌ User ID tidak valid. Silakan login ulang.")
        return 0
    
    # Required columns
    required_cols = ['order_date', 'Final_Price']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Kolom wajib tidak ditemukan: {', '.join(missing_cols)}")
        return 0
    
    # Map DataFrame columns to database columns
    db_column_map = {
        'order_date': 'order_date',
        'invoice_date': 'invoice_date',
        'Final_Price': 'Final_Price',
        'qty': 'qty',
        'price': 'price',
        'bundle_price': 'bundle_price',
        'subtotal': 'subtotal',
        'discount': 'discount',
        'shipping_fee': 'shipping_fee',
        'order_number': 'order_number',
        'name': 'name',
        'brand': 'brand',
        'product_name': 'product_name',
        'sku': 'sku',
        'Cat2': 'Cat2',
        'kode_voucher': 'kode_voucher',
        'status': 'status',
        'payment_method': 'payment_method',
        'used_point': 'used_point',
        'diskonproposional': 'diskonproposional',
        'HP': 'HP',
        'bulan': 'bulan',
        'year': 'year',
        'month': 'month',
        'month_name': 'month_name'
    }
    
    # Get available columns
    available_cols = [col for col in db_column_map.keys() if col in df.columns]
    db_cols = ['user_id'] + [db_column_map[col] for col in available_cols]
    
    # ✅ PostgreSQL uses %s placeholders (sama seperti MySQL)
    placeholders = ", ".join(["%s"] * len(db_cols))
    cols_sql = ", ".join([f'"{c}"' for c in db_cols])  # ✅ Use double quotes for PostgreSQL
    insert_sql = f"INSERT INTO transactions ({cols_sql}) VALUES ({placeholders})"
    
    conn = None
    inserted = 0
    skipped = 0
    
    try:
        conn = get_connection()
        
        if batch_size is None or batch_size <= 0:
            batch_size = 1000
        
        # Process in batches
        total_rows = len(df)
        total_batches = (total_rows - 1) // batch_size + 1
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            values = []
            
            for _, row in batch_df.iterrows():
                row_vals = [int(user_id)]
                skip_row = False
                
                for col in available_cols:
                    v = row[col]
                    
                    if col in required_cols and pd.isna(v):
                        skipped += 1
                        skip_row = True
                        break
                    
                    if pd.isna(v):
                        row_vals.append(None)
                    elif isinstance(v, (pd.Timestamp, datetime)):
                        row_vals.append(v.strftime("%Y-%m-%d %H:%M:%S"))
                    elif isinstance(v, (int, np.integer)):
                        row_vals.append(int(v))
                    elif isinstance(v, (float, np.floating)):
                        row_vals.append(float(v))
                    else:
                        row_vals.append(str(v))
                
                if not skip_row:
                    values.append(tuple(row_vals))
            
            # Insert batch using executemany
            if values:
                with conn.cursor() as cur:
                    # ✅ PostgreSQL executemany works the same way
                    cur.executemany(insert_sql, values)
                conn.commit()
                inserted += len(values)
            
            # Update progress
            progress = (batch_num + 1) / total_batches
            progress_bar.progress(progress)
            status_text.text(f"💾 Saving batch {batch_num + 1}/{total_batches} - Inserted: {inserted:,}")
        
        progress_bar.empty()
        status_text.empty()
        
        if skipped > 0:
            st.warning(f"⚠️ {skipped:,} baris dilewati")
        
        st.success(f"✅ Berhasil menyimpan {inserted:,} transaksi ke database!")
        
        return inserted
        
    except psycopg2.errors.UniqueViolation as e:  # ✅ PostgreSQL exception
        if conn:
            conn.rollback()
        st.error(f"❌ Data duplikat: {e}")
        return inserted
        
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"❌ Failed to save: {e}")
        import traceback
        with st.expander("🔍 Debug Info"):
            st.code(traceback.format_exc())
        return inserted
        
    finally:
        if conn:
            conn.close()


def test_database_connection():
    """
    Test koneksi ke Supabase PostgreSQL
    """
    conn = None
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            # Test query
            cursor.execute("SELECT 1")
            
            # Get database info
            cursor.execute("SELECT current_database()")
            db_name = cursor.fetchone()['current_database']
            
            # Get table list
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cursor.fetchall()
            
            # Get transactions table structure
            cursor.execute("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'transactions'
                ORDER BY ordinal_position
            """)
            structure = cursor.fetchall()
        
        st.success("✅ Koneksi Supabase berhasil!")
        st.info(f"📊 Database: {db_name}")
        st.write("📋 Available tables:", [t['table_name'] for t in tables])
        
        if structure:
            with st.expander("🔍 Transactions Table Structure"):
                st.dataframe(pd.DataFrame(structure))
        else:
            st.warning("⚠️ Table 'transactions' belum dibuat")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Koneksi gagal: {e}")
        st.info("💡 Periksa: Supabase running, database created, credentials benar")
        return False
        
    finally:
        if conn:
            conn.close()

def load_transactions_from_db(user_id: int, limit=None, offset=0, date_from=None, date_to=None):
    """
    Load transactions from PostgreSQL database
    """
    if user_id is None:
        st.error("❌ User ID tidak ditemukan. Silakan login ulang.")
        return None
    
    conn = None
    try:
        conn = get_connection()
        
        # Build query
        base_query = """
            SELECT 
                id,
                order_number,
                order_date,
                invoice_date,
                name,
                brand,
                product_name,
                sku,
                "Cat2",
                qty,
                price,
                bundle_price,
                subtotal,
                discount,
                shipping_fee,
                "Final_Price",
                bulan,
                year,
                month,
                created_at
            FROM transactions 
            WHERE user_id = %s
        """
        
        params = [user_id]
        
        if date_from:
            base_query += " AND order_date >= %s"
            params.append(date_from)
        
        if date_to:
            base_query += " AND order_date <= %s"
            params.append(date_to)
        
        base_query += " ORDER BY order_date DESC"
        
        if limit is not None:
            base_query += " LIMIT %s OFFSET %s"
            params.extend([limit, offset])
        
        # Get total count
        count_query = "SELECT COUNT(*) as total FROM transactions WHERE user_id = %s"
        count_params = [user_id]
        
        if date_from:
            count_query += " AND order_date >= %s"
            count_params.append(date_from)
        if date_to:
            count_query += " AND order_date <= %s"
            count_params.append(date_to)
        
        with conn.cursor() as cursor:
            # Get total count
            cursor.execute(count_query, count_params)
            total_rows = cursor.fetchone()['total']
            
            if total_rows == 0:
                st.warning(f"⚠️ Tidak ada data transaksi untuk user_id: {user_id}")
                return None
            
            # Show loading progress for large datasets
            if total_rows > 50000:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text(f"📥 Loading {total_rows:,} rows...")
            
            # Load data
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            
            if total_rows > 50000:
                progress_bar.progress(0.5)
                status_text.text(f"🔄 Converting to DataFrame...")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        if total_rows > 50000:
            progress_bar.progress(0.7)
            status_text.text(f"⚙️ Optimizing memory...")
        
        # Convert date columns
        date_columns = ['order_date', 'invoice_date', 'created_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Optimize memory
        df = optimize_dataframe_memory(df)
        
        if total_rows > 50000:
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
        
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.success(f"✅ Loaded {len(df):,} rows ({memory_mb:.2f} MB)")
        
        return df
        
    except psycopg2.OperationalError as e:  # ✅ PostgreSQL exception
        st.error(f"❌ Database connection error: {e}")
        return None
        
    except psycopg2.ProgrammingError as e:  # ✅ PostgreSQL exception
        st.error(f"❌ SQL query error: {e}")
        return None
        
    except Exception as e:
        st.error(f"❌ Failed to load transactions: {e}")
        import traceback
        with st.expander("🔍 Debug Info"):
            st.code(traceback.format_exc())
        return None
        
    finally:
        if conn:
            conn.close()


def load_transactions_summary(user_id: int):
    """
    Load summary statistics without loading all data
    Useful for dashboard metrics
    """
    if user_id is None:
        return None
    
    conn = None
    try:
        conn = get_connection()
        
        query = """
            SELECT 
                COUNT(*) as total_transactions,
                SUM(Final_Price) as total_revenue,
                AVG(Final_Price) as avg_transaction,
                MIN(order_date) as first_transaction,
                MAX(order_date) as last_transaction,
                COUNT(DISTINCT DATE(order_date)) as unique_days
            FROM transactions 
            WHERE user_id = %s
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
        
        return result
        
    except Exception as e:
        st.error(f"❌ Failed to load summary: {e}")
        return None
        
    finally:
        if conn:
            conn.close()


def load_transactions_by_date_range(user_id: int, start_date, end_date):
    """
    Load transactions for a specific date range
    More efficient for time-series analysis
    """
    if user_id is None:
        return None
    
    conn = None
    try:
        conn = get_connection()
        
        query = """
            SELECT 
                DATE(order_date) as date,
                SUM(Final_Price) as daily_revenue,
                COUNT(*) as daily_transactions,
                AVG(Final_Price) as avg_price
            FROM transactions 
            WHERE user_id = %s 
            AND order_date BETWEEN %s AND %s
            GROUP BY DATE(order_date)
            ORDER BY date ASC
        """
        
        df = pd.read_sql(query, conn, params=(user_id, start_date, end_date))
        
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            st.success(f"✅ Loaded aggregated data for {len(df)} days")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to load date range: {e}")
        return None
        
    finally:
        if conn:
            conn.close()

def save_forecasts_to_db(user_id: int, model: str, dates, values):
    """
    Save forecasts to PostgreSQL database
    """
    if user_id is None:
        st.error("User tidak ditemukan. Login terlebih dahulu.")
        return 0
    
    insert_sql = """
        INSERT INTO forecasts 
        (user_id, username, model_type, forecast_date, forecast_value) 
        VALUES (%s, %s, %s, %s, %s)
    """
    payload = []
    username = st.session_state.get('username', 'unknown')
    
    for d, v in zip(dates, values):
        if isinstance(d, (pd.Timestamp, datetime)):
            d_val = d.date().isoformat()
        else:
            d_val = str(d)
        payload.append((user_id, username, model, d_val, float(v)))
    
    conn = get_connection()
    inserted = 0
    try:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, payload)
        conn.commit()
        inserted = len(payload)
    except Exception as e:
        conn.rollback()
        st.error(f"Failed to save forecasts: {e}")
    finally:
        conn.close()
    return inserted

DEBUG_MODE = False  # ← Set ke True dulu untuk debugging

def get_forecast_history(user_id: int, limit=20):
    """
    Get forecast history grouped by session - PostgreSQL version
    """
    if user_id is None:
        return None
    
    conn = None
    try:
        conn = get_connection()
        
        query = """
            SELECT 
                model_type,
                DATE(created_at) AS session_date,
                MIN(forecast_date) AS start_date,
                MAX(forecast_date) AS end_date,
                COUNT(*) AS forecast_days,
                AVG(forecast_value) AS avg_forecast,
                MIN(forecast_value) AS min_forecast,
                MAX(forecast_value) AS max_forecast,
                SUM(forecast_value) AS total_forecast
            FROM forecasts
            WHERE user_id = %s
            GROUP BY model_type, DATE(created_at)
            ORDER BY DATE(created_at) DESC 
            LIMIT %s
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, (user_id, limit))
            results = cursor.fetchall()
        
        if not results or len(results) == 0:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Convert numeric columns
        df['forecast_days'] = pd.to_numeric(df['forecast_days'], errors='coerce').fillna(0).astype(int)
        df['avg_forecast'] = pd.to_numeric(df['avg_forecast'], errors='coerce').fillna(0.0)
        df['min_forecast'] = pd.to_numeric(df['min_forecast'], errors='coerce').fillna(0.0)
        df['max_forecast'] = pd.to_numeric(df['max_forecast'], errors='coerce').fillna(0.0)
        df['total_forecast'] = pd.to_numeric(df['total_forecast'], errors='coerce').fillna(0.0)
        
        # Format date columns
        for col in ['session_date', 'start_date', 'end_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].dt.strftime('%Y-%m-%d')
                df[col] = df[col].fillna('N/A')
        
        # Format numeric columns
        df['avg_forecast'] = df['avg_forecast'].apply(lambda x: f"{x:,.2f}")
        df['min_forecast'] = df['min_forecast'].apply(lambda x: f"{x:,.2f}")
        df['max_forecast'] = df['max_forecast'].apply(lambda x: f"{x:,.2f}")
        df['total_forecast'] = df['total_forecast'].apply(lambda x: f"{x:,.2f}")
        
        # Rename columns
        df = df.rename(columns={
            'model_type': 'Model',
            'session_date': 'Session Date',
            'start_date': 'Forecast Start',
            'end_date': 'Forecast End',
            'forecast_days': 'Days',
            'avg_forecast': 'Avg Value',
            'min_forecast': 'Min Value',
            'max_forecast': 'Max Value',
            'total_forecast': 'Total Value'
        })
        
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to load forecast history: {e}")
        import traceback
        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc())
        return pd.DataFrame()
        
    finally:
        if conn:
            conn.close()


def get_forecast_details(user_id: int, session_date: str, model_type: str):
    """
    Get detailed daily forecast values - PostgreSQL version
    """
    if user_id is None:
        return None
    
    conn = None
    try:
        conn = get_connection()
        
        query = """
            SELECT 
                forecast_date,
                forecast_value
            FROM forecasts 
            WHERE user_id = %s 
            AND DATE(created_at) = %s
            AND model_type = %s
            ORDER BY forecast_date ASC
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, (user_id, session_date, model_type))
            results = cursor.fetchall()
        
        if not results or len(results) == 0:
            return pd.DataFrame(columns=['Date', 'Forecast Value'])
        
        df = pd.DataFrame(results)
        
        # Convert to proper types
        df['forecast_date'] = pd.to_datetime(df['forecast_date'], errors='coerce')
        df['forecast_value'] = pd.to_numeric(df['forecast_value'], errors='coerce')
        
        # Remove invalid rows
        df = df.dropna()
        
        if len(df) == 0:
            return pd.DataFrame(columns=['Date', 'Forecast Value'])
        
        # Rename columns
        df = df.rename(columns={
            'forecast_date': 'Date',
            'forecast_value': 'Forecast Value'
        })
        
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to load forecast details: {e}")
        import traceback
        with st.expander("🔍 Debug Info"):
            st.code(traceback.format_exc())
        return None
        
    finally:
        if conn:
            conn.close()


def get_latest_forecast_summary(user_id: int):
    """
    Get summary of most recent forecast - PostgreSQL version
    """
    if user_id is None:
        return None
    
    conn = None
    try:
        conn = get_connection()
        
        query = """
            SELECT 
                model_type,
                COUNT(*) as forecast_days,
                AVG(forecast_value) as avg_value,
                SUM(forecast_value) as total_value,
                MIN(forecast_date) as start_date,
                MAX(forecast_date) as end_date,
                DATE(created_at) as session_date,
                created_at
            FROM forecasts 
            WHERE user_id = %s 
            GROUP BY model_type, created_at
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
        
        if not result:
            return None
        
        # Convert values to proper types
        result['forecast_days'] = int(result['forecast_days']) if result['forecast_days'] else 0
        result['avg_value'] = float(result['avg_value']) if result['avg_value'] else 0.0
        result['total_value'] = float(result['total_value']) if result['total_value'] else 0.0
        
        # Convert dates to string
        if result.get('session_date'):
            if isinstance(result['session_date'], datetime):
                result['session_date'] = result['session_date'].strftime('%Y-%m-%d')
            else:
                result['session_date'] = str(result['session_date'])
        
        return result
        
    except Exception as e:
        st.error(f"❌ Failed to load latest forecast: {e}")
        return None
        
    finally:
        if conn:
            conn.close()




def get_user_statistics(user_id: int):
    """
    Get statistics for a specific user - PostgreSQL version
    """
    if user_id is None:
        return {'total_transactions': 0, 'total_forecasts': 0, 'latest_upload': None}
    
    conn = None
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            # Total transactions
            cursor.execute(
                "SELECT COUNT(*) as count FROM transactions WHERE user_id = %s", 
                (user_id,)
            )
            result = cursor.fetchone()
            total_transactions = result['count'] if result else 0
            
            # Total forecasts
            cursor.execute(
                """
                SELECT COUNT(*) AS count
                FROM (
                    SELECT 
                        model_type,
                        DATE(created_at) AS session_date
                    FROM forecasts
                    WHERE user_id = %s
                    GROUP BY model_type, DATE(created_at)
                    LIMIT 999
                ) AS t
                """, 
                (user_id,)
            )
            result = cursor.fetchone()
            total_forecasts = result['count'] if result else 0
            
            # Latest upload
            cursor.execute(
                "SELECT MAX(created_at) as latest FROM transactions WHERE user_id = %s", 
                (user_id,)
            )
            result = cursor.fetchone()
            latest_upload = result['latest'] if result and result['latest'] else None
        
        return {
            'total_transactions': int(total_transactions) if total_transactions else 0,
            'total_forecasts': int(total_forecasts) if total_forecasts else 0,
            'latest_upload': latest_upload
        }
        
    except psycopg2.OperationalError as e:
        st.error(f"❌ Database connection error: {e}")
        return {'total_transactions': 0, 'total_forecasts': 0, 'latest_upload': None}
        
    except Exception as e:
        st.error(f"❌ Failed to get user statistics: {e}")
        return {'total_transactions': 0, 'total_forecasts': 0, 'latest_upload': None}
        
    finally:
        if conn:
            conn.close()

def is_admin_user(user_id):
    """Check if user is admin - PostgreSQL version"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT is_admin FROM users WHERE id = %s", (user_id,))
            result = cursor.fetchone()
            return result and result['is_admin'] == 1
    except Exception as e:
        st.error(f"Error checking admin status: {e}")
        return False
    finally:
        conn.close()

def get_all_users_admin():
    """
    Get all users with their statistics - PostgreSQL version
    """
    conn = None
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            # Simplified query
            query = """
                SELECT 
                    u.id,
                    u.username,
                    u.raw_password,
                    u.is_admin,
                    u.created_at,
                    COUNT(DISTINCT t.id) as total_transactions,
                    0 as total_forecasts
                FROM users u
                LEFT JOIN transactions t ON u.id = t.user_id
                WHERE u.is_admin = 0
                GROUP BY u.id, u.username, u.raw_password, u.is_admin, u.created_at
                ORDER BY u.created_at DESC
                LIMIT 100
            """
            cursor.execute(query)
            results = cursor.fetchall()
            
            if results:
                df = pd.DataFrame(results)
                
                # Get forecast counts separately
                for idx, row in df.iterrows():
                    cursor.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM (
                            SELECT 
                                model_type,
                                DATE(created_at) AS session_date
                            FROM forecasts
                            WHERE user_id = %s
                            GROUP BY model_type, DATE(created_at)
                            LIMIT 999
                        ) AS t
                        """,
                        (row['id'],)
                    )
                    fc_result = cursor.fetchone()
                    df.at[idx, 'total_forecasts'] = fc_result['count'] if fc_result else 0
                
                return df
            else:
                return pd.DataFrame()
                
    except Exception as e:
        st.error(f"❌ Error loading users: {e}")
        return pd.DataFrame()
        
    finally:
        if conn:
            conn.close()




def debug_upload_history():
    """
    Debug function to check upload_history table
    Call this from admin dashboard to diagnose issues
    """
    st.subheader("🔍 Upload History Debug")
    
    conn = None
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            # Check table structure
            st.write("**1. Table Structure:**")
            cursor.execute("DESCRIBE upload_history")
            structure = cursor.fetchall()
            st.dataframe(pd.DataFrame(structure))
            
            # Check total records
            st.write("**2. Total Records:**")
            cursor.execute("SELECT COUNT(*) as total FROM upload_history")
            total = cursor.fetchone()['total']
            st.metric("Total Records", total)
            
            # Check recent records
            st.write("**3. Recent Records (Last 5):**")
            cursor.execute("""
                SELECT * FROM upload_history 
                ORDER BY upload_date DESC 
                LIMIT 5
            """)
            recent = cursor.fetchall()
            if recent:
                st.dataframe(pd.DataFrame(recent))
            else:
                st.info("No records found")
            
            # Check by user
            st.write("**4. Records by User:**")
            cursor.execute("""
                SELECT 
                    u.username,
                    COUNT(*) as upload_count
                FROM upload_history uh
                LEFT JOIN users u ON uh.user_id = u.id
                GROUP BY u.username
            """)
            by_user = cursor.fetchall()
            if by_user:
                st.dataframe(pd.DataFrame(by_user))
            else:
                st.info("No user data found")
                
    except Exception as e:
        st.error(f"Debug error: {e}")
        import traceback
        st.code(traceback.format_exc())
        
    finally:
        if conn:
            conn.close()







def get_latest_uploads_admin():
    """
    Get latest CSV uploads per user - PostgreSQL version
    """
    conn = None
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            # Check total records
            cursor.execute("SELECT COUNT(*) as total FROM upload_history")
            total_result = cursor.fetchone()
            total_records = total_result['total'] if total_result else 0
            
            st.info(f"📊 Total records in upload_history: {total_records}")
            
            if total_records == 0:
                st.warning("⚠️ No upload history records found!")
                return pd.DataFrame()
            
            # Main query
            query = """
                SELECT 
                    uh.id,
                    uh.user_id,
                    u.username,
                    uh.filename,
                    uh.file_size_mb,
                    uh.total_rows,
                    uh.upload_date,
                    uh.status,
                    uh.error_message
                FROM upload_history uh
                LEFT JOIN users u ON uh.user_id = u.id
                ORDER BY uh.upload_date DESC
                LIMIT 100
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            st.success(f"✅ Query returned {len(results)} records")
        
        if results and len(results) > 0:
            df = pd.DataFrame(results)
            df['username'] = df['username'].fillna('Unknown User')
            return df
        else:
            st.warning("⚠️ Query returned no results")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ Error loading upload history: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()
        
    finally:
        if conn:
            conn.close()

def get_user_latest_upload(user_id):
    """Get latest upload for specific user"""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            query = """
                SELECT 
                    filename,
                    file_size_mb,
                    total_rows,
                    upload_date
                FROM upload_history
                WHERE user_id = %s
                ORDER BY upload_date DESC
                LIMIT 1
            """
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
        return result
    except Exception as e:
        # Silent fail untuk fungsi ini (tidak kritis)
        return None
    finally:
        if conn:
            conn.close()

def log_upload_history(user_id, username, filename, file_size_mb, total_rows, status='success', error_msg=None):
    """
    Log CSV upload to history table - PostgreSQL version
    ✅ MENGGUNAKAN RETURNING untuk mendapatkan ID yang baru diinsert
    """
    import traceback
    
    conn = None
    
    try:
        st.write("🔍 **Validating data...**")
        
        if user_id is None or user_id <= 0:
            st.error(f"❌ Invalid user_id: {user_id}")
            return False
        
        if not username or not filename:
            st.error(f"❌ Missing required fields")
            return False
        
        st.write(f"✓ user_id: {user_id}")
        st.write(f"✓ username: {username}")
        st.write(f"✓ filename: {filename}")
        
        st.write("🔌 **Connecting to database...**")
        conn = get_connection()
        st.write("✓ Connected")
        
        st.write("🔍 **Verifying upload_history table...**")
        
        with conn.cursor() as cursor:
            # Check if table exists
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'upload_history'
            """)
            table_exists = cursor.fetchone()
            
            if not table_exists:
                st.error("❌ Table 'upload_history' does not exist!")
                return False
            
            st.write("✓ Table exists")
        
        st.write("💾 **Inserting record...**")
        
        with conn.cursor() as cursor:
            # ✅ PostgreSQL: Use RETURNING untuk get last insert ID
            insert_query = """
                INSERT INTO upload_history 
                (user_id, username, filename, file_size_mb, total_rows, status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            
            values = (
                int(user_id),
                str(username),
                str(filename),
                float(file_size_mb),
                int(total_rows),
                str(status),
                str(error_msg) if error_msg else None
            )
            
            cursor.execute(insert_query, values)
            last_id = cursor.fetchone()['id']  # ✅ Get ID dari RETURNING
            
            st.write("✓ INSERT executed")
        
        st.write("💾 **Committing transaction...**")
        conn.commit()
        st.write("✓ Committed")
        
        st.write("🔍 **Verifying insert...**")
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM upload_history 
                WHERE id = %s
            """, (last_id,))
            
            saved_record = cursor.fetchone()
            
            if saved_record:
                st.success("✅ **Record verified in database!**")
                with st.expander("👁️ View Saved Record"):
                    st.json(saved_record)
                return True
            else:
                st.error("❌ Record not found after insert!")
                return False
        
    except psycopg2.errors.UniqueViolation as e:  # ✅ PostgreSQL exception
        if conn:
            conn.rollback()
        st.error(f"❌ Duplicate key error: {e}")
        return False
        
    except psycopg2.OperationalError as e:  # ✅ PostgreSQL exception
        if conn:
            conn.rollback()
        st.error(f"❌ Database connection error: {e}")
        return False
        
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"❌ Unexpected error: {e}")
        with st.expander("🔍 Full Error Traceback"):
            st.code(traceback.format_exc())
        return False
        
    finally:
        if conn:
            conn.close()
            st.write("🔌 Database connection closed")


def verify_upload_history_count():
    """
    Quick check to verify upload history records vs actual transactions
    Add this as a button in your admin dashboard
    """
    st.subheader("🔍 Quick Verification")
    
    conn = None
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            # Count upload history records
            cursor.execute("SELECT COUNT(*) as count FROM upload_history")
            upload_count = cursor.fetchone()['count']
            
            # Count distinct uploads by checking transactions grouped by user and date
            cursor.execute("""
                SELECT COUNT(DISTINCT CONCAT(user_id, '-', DATE(created_at))) as count
                FROM transactions
            """)
            transaction_dates = cursor.fetchone()['count']
            
            # Show comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Upload History Records", upload_count)
            
            with col2:
                st.metric("Distinct Upload Dates in Transactions", transaction_dates)
            
            if upload_count < transaction_dates:
                st.warning(f"⚠️ Missing {transaction_dates - upload_count} upload records!")
                st.info("💡 Some uploads were not logged to upload_history table")
            else:
                st.success("✅ All uploads are recorded!")
                
    except Exception as e:
        st.error(f"Error: {e}")
        
    finally:
        if conn:
            conn.close()



def reset_user_password_admin(user_id, new_password):
    """Admin reset user password - PostgreSQL version"""
    conn = get_connection()
    try:
        hashed_pw = hash_password(new_password)
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET password = %s, raw_password = %s WHERE id = %s",
                (hashed_pw, new_password, user_id)
            )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Failed to reset password: {e}")
        return False
    finally:
        conn.close()

def delete_user_admin(user_id):
    """Admin delete user - PostgreSQL version"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Failed to delete user: {e}")
        return False
    finally:
        conn.close()

def save_transactions_to_db_with_logging(df, user_id, filename, batch_size=1000):
    """
    Save transactions with upload history logging - GUARANTEED LOGGING
    
    CRITICAL: This function will ALWAYS log to upload_history, even if save fails
    """
    import traceback
    
    # Validate inputs
    if df is None or len(df) == 0:
        st.error("❌ Data kosong, tidak ada yang disimpan")
        return 0
    
    if not filename:
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Calculate metadata
    file_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    total_rows = len(df)
    username = st.session_state.get('username', 'unknown')
    
    # Show upload info
    st.info(f"📝 **Upload Info:**")
    st.write(f"- User: **{username}** (ID: {user_id})")
    st.write(f"- File: **{filename}**")
    st.write(f"- Size: **{file_size_mb:.2f} MB**")
    st.write(f"- Rows: **{total_rows:,}**")
    st.markdown("---")
    
    # Initialize tracking variables
    inserted = 0
    upload_status = 'failed'
    error_message = None
    
    try:
        # ========== STEP 1: SAVE TO TRANSACTIONS TABLE ==========
        st.info("💾 Step 1/2: Saving to transactions table...")
        progress_placeholder = st.empty()
        
        inserted = save_transactions_to_db(df, user_id, batch_size)
        
        if inserted > 0:
            upload_status = 'success'
            progress_placeholder.success(f"✅ Saved {inserted:,} rows to transactions table")
        else:
            upload_status = 'failed'
            error_message = "No rows inserted - check data validation"
            progress_placeholder.warning(f"⚠️ No rows saved to transactions table")
            
    except Exception as save_error:
        upload_status = 'failed'
        error_message = str(save_error)
        st.error(f"❌ Save error: {error_message}")
        
        # Show detailed error for debugging
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
    
    finally:
        # ========== STEP 2: LOG TO UPLOAD_HISTORY TABLE ==========
        # ✅ CRITICAL: ALWAYS execute this, even if save failed
        st.info("📝 Step 2/2: Logging to upload_history table...")
        
        try:
            # Call log function
            log_success = log_upload_history(
                user_id=user_id,
                username=username,
                filename=filename,
                file_size_mb=file_size_mb,
                total_rows=inserted if inserted > 0 else 0,  # Log actual inserted rows
                status=upload_status,
                error_msg=error_message
            )
            
            if log_success:
                st.success("✅ Upload history logged successfully!")
            else:
                st.error("❌ Failed to log upload history!")
                st.warning("⚠️ Your data was saved but upload history was not recorded")
                
        except Exception as log_error:
            st.error(f"❌ Logging error: {log_error}")
            st.warning("⚠️ Your data was saved but upload history failed to record")
            
            with st.expander("🔍 Logging Error Details"):
                st.code(traceback.format_exc())
        
        # ========== STEP 3: SHOW FINAL SUMMARY ==========
        st.markdown("---")
        st.subheader("📊 Upload Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_icon = "✅" if upload_status == 'success' else "❌"
            st.metric("Save Status", f"{status_icon} {upload_status.upper()}")
        
        with col2:
            st.metric("Rows Saved", f"{inserted:,}")
        
        with col3:
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        
        with col4:
            log_icon = "✅" if log_success else "❌"
            st.metric("History Logged", log_icon)
    
    return inserted
# ------------------ Advanced Metrics ------------------
class AdvancedMetrics:
    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def smape(y_true, y_pred):
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        return np.mean(np.abs(y_true - y_pred) / denominator) * 100
    
    @staticmethod
    def mase(y_true, y_pred, y_train, seasonality=1):
        if seasonality > 1 and len(y_train) > seasonality:
            naive_forecast = y_train[:-seasonality]
            naive_actual = y_train[seasonality:]
            mae_naive = np.mean(np.abs(naive_actual - naive_forecast))
        else:
            mae_naive = np.mean(np.abs(np.diff(y_train)))
        if mae_naive == 0:
            mae_naive = 1e-8
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        return mae_forecast / mae_naive

# ------------------ Data cleaning & helpers ------------------
def remove_outliers_iqr(data, columns):
    cleaned = data.copy()
    outlier_info = []
    for col in columns:
        if col in cleaned.columns:
            Q1 = cleaned[col].quantile(0.25)
            Q3 = cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            before = len(cleaned)
            cleaned = cleaned[(cleaned[col] >= lower_bound) & (cleaned[col] <= upper_bound)]
            after = len(cleaned)
            outlier_info.append({
                'column': col,
                'removed': before - after,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
    return cleaned, outlier_info

def clean_data_advanced(df):
    cleaning_log = {
        'original_rows': len(df),
        'steps': []
    }
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        cleaning_log['steps'].append('✅ Converted order_date to datetime')
    if 'invoice_date' in df.columns:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
        cleaning_log['steps'].append('✅ Converted invoice_date to datetime')
    num_cols = ["qty", "price", "bundle_price", "subtotal", "discount", 
                "shipping_fee", "used_point", "diskonproposional", "Final_Price",
                "bulan", "year", "month"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    cleaning_log['steps'].append(f'✅ Converted {len([c for c in num_cols if c in df.columns])} columns to numeric')
    before_dup = len(df)
    df = df.drop_duplicates()
    after_dup = len(df)
    cleaning_log['steps'].append(f'✅ Removed {before_dup - after_dup} duplicates')
    outlier_cols = ["qty", "price", "subtotal", "discount", "Final_Price"]
    available_outlier_cols = [col for col in outlier_cols if col in df.columns]
    if available_outlier_cols:
        df, outlier_info = remove_outliers_iqr(df, available_outlier_cols)
        cleaning_log['outlier_info'] = outlier_info
        total_outliers = sum([info['removed'] for info in outlier_info])
        cleaning_log['steps'].append(f'✅ Removed {total_outliers} outliers using IQR method')
    cleaning_log['final_rows'] = len(df)
    cleaning_log['total_removed'] = cleaning_log['original_rows'] - cleaning_log['final_rows']
    return df, cleaning_log

# ------------------ SARIMA & LSTM classes ------------------
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class SARIMAAutoForecaster:
    def __init__(self):
        self.model = None
        self.model_params = None
        self.fitted = False
        self.search_results = []
    
    def auto_fit(self, train_data, max_p=3, max_d=2, max_q=3, 
                 max_P=2, max_D=1, max_Q=2, seasonal_period=7):
        import itertools
        best_aic = np.inf
        best_params = None
        search_log = []
        p_range = range(1, max_p + 1)
        d_range = range(0, max_d + 1)
        q_range = range(1, max_q + 1)
        P_range = range(0, max_P + 1)
        D_range = range(0, max_D + 1)
        Q_range = range(0, max_Q + 1)
        tested = 0
        successful = 0
        for p, d, q, P, D, Q in itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range):
            tested += 1
            if p == 0 and q == 0:
                continue
            try:
                model = SARIMAX(
                    train_data,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted_model = model.fit(disp=False, maxiter=100, method='lbfgs')
                search_log.append({
                    'order': (p, d, q),
                    'seasonal_order': (P, D, Q, seasonal_period),
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'status': 'success'
                })
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_params = {
                        'order': (p, d, q),
                        'seasonal_order': (P, D, Q, seasonal_period),
                        'model': fitted_model,
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic,
                        'loglik': fitted_model.llf
                    }
                successful += 1
            except Exception as e:
                search_log.append({
                    'order': (p, d, q),
                    'seasonal_order': (P, D, Q, seasonal_period),
                    'status': 'failed',
                    'error': str(e)
                })
                continue
        if best_params is None:
            raise Exception("Tidak ada model SARIMA yang berhasil di-fit")
        self.model = best_params['model']
        self.model_params = best_params
        self.fitted = True
        self.search_results = search_log
        summary = {
            'best_order': best_params['order'],
            'best_seasonal_order': best_params['seasonal_order'],
            'best_aic': best_params['aic'],
            'best_bic': best_params['bic'],
            'best_loglik': best_params['loglik'],
            'total_tested': tested,
            'total_successful': successful,
            'success_rate': (successful / tested) * 100 if tested > 0 else 0
        }
        return summary, search_log
    
    def forecast(self, steps):
        if not self.fitted:
            raise Exception("Model belum di-fit")
        forecast_result = self.model.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean.values
        forecast_ci = forecast_result.conf_int().values
        return forecast_mean, forecast_ci[:, 0], forecast_ci[:, 1]
    
    def get_model_summary(self):
        if not self.fitted:
            return None
        return {
            'order': self.model_params['order'],
            'seasonal_order': self.model_params['seasonal_order'],
            'aic': self.model_params['aic'],
            'bic': self.model_params['bic'],
            'loglik': self.model_params['loglik'],
            'summary_text': str(self.model.summary())
        }
    
    def plot_diagnostics(self):
        if not self.fitted:
            return None
        fig = self.model.plot_diagnostics(figsize=(15, 10))
        return fig

class LSTMForecaster:
    def __init__(self, sequence_length=30, lstm_units=50):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.model = None
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        self.fitted = False
    
    def _create_sequences(self, data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def fit(self, train_data, validation_split=0.2, epochs=100, batch_size=32):
        """Fixed: Import keras dengan fallback"""
        train_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
        X, y = self._create_sequences(train_scaled, self.sequence_length)
        
        if len(X) < 50:
            raise Exception(f"Not enough data for LSTM. Need at least {self.sequence_length + 50} points, got {len(X)} sequences")
        
        # ✅ FIX: Try standalone keras first, fallback to tensorflow.keras
        try:
            from keras.models import Sequential
            from keras.layers import LSTM, Dense, Dropout
            from keras.optimizers import Adam
            from keras.callbacks import EarlyStopping
        except ImportError:
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                from tensorflow.keras.optimizers import Adam
                from tensorflow.keras.callbacks import EarlyStopping
            except ImportError:
                raise Exception("❌ Keras/TensorFlow not available. Install: pip install tensorflow")
        
        # Build model
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(self.lstm_units//2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        
        history = model.fit(
            X_reshaped, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.model = model
        self.fitted = True
        
        return history
    
    def forecast(self, train_data, steps, add_noise=True):
        if not self.fitted:
            raise Exception("Model not fitted yet")
        
        last_sequence = train_data.tail(self.sequence_length).values
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        forecasts = []
        current_sequence = last_sequence_scaled.copy()
        train_std = train_data.std() * 0.15
        
        for _ in range(steps):
            X_input = current_sequence.reshape(1, self.sequence_length, 1)
            next_pred_scaled = self.model.predict(X_input, verbose=0)[0, 0]
            next_pred = self.scaler.inverse_transform([[next_pred_scaled]])[0, 0]
            
            if add_noise:
                noise = np.random.normal(0, train_std)
                next_pred = next_pred + noise
                next_pred = max(0, next_pred)
            
            forecasts.append(next_pred)
            next_pred_scaled_noisy = self.scaler.transform([[next_pred]])[0, 0]
            current_sequence = np.append(current_sequence[1:], next_pred_scaled_noisy)
        
        return np.array(forecasts)

# ------------------ Data chunk & load helpers ------------------
def generate_sample_data_chunked(n=1000000, chunk_size=100000):
    chunks = []
    for i in range(0, n, chunk_size):
        size = min(chunk_size, n - i)
        chunk = pd.DataFrame({
            'id': range(i + 1, i + size + 1),
            'order_date': pd.date_range(start='2023-01-01', periods=size, freq='H'),
            'qty': np.random.randint(1, 10, size),
            'price': np.random.randint(10000, 100000, size),
            'Final_Price': np.random.randint(10000, 100000, size),
            'category': np.random.choice(['A', 'B', 'C', 'D'], size)
        })
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def load_csv_chunked(file, chunksize=100000):
    chunks = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        file.seek(0)
        reader = pd.read_csv(file, chunksize=chunksize, low_memory=False)
        total_rows = 0
        chunk_count = 0
        for i, chunk in enumerate(reader):
            chunks.append(chunk)
            total_rows += len(chunk)
            chunk_count += 1
            status_text.text(f"Loading chunk {chunk_count}... Total: {total_rows:,} rows")
            progress = min((chunk_count * 5) / 100, 0.95)
            progress_bar.progress(progress)
            if chunk_count % 10 == 0:
                gc.collect()
        progress_bar.progress(0.98)
        status_text.text(f"Combining {chunk_count} chunks...")
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        progress_bar.progress(1.0)
        status_text.text(f"✅ Loaded {total_rows:,} rows successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None
    finally:
        progress_bar.empty()
        status_text.empty()

# ------------------ UI: Login & Register dengan Change Password ------------------
def validate_password(password):
    """Validate password requirements"""
    import re
    if len(password) < 4:
        return False, "❌ Password minimal 4 karakter"
    
    if not re.search(r"[A-Z]", password):
        return False, "❌ Password harus mengandung minimal 1 huruf besar"
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>_\-+=\[\]\\;'/`~]", password):
        return False, "❌ Password harus mengandung minimal 1 simbol (!@#$%^&* dll)"
    
    return True, "✅ Password valid"

def login_page():
    st.title("PT. XYZ")
    tab1, tab2, tab3 = st.tabs(["📝 Register", "🔐 Login", "🔑 Change Password"])
    
    # TAB 1: REGISTER
    with tab1:
        st.subheader("Register New Account")
        st.info("📋 **Password Requirements:**\n- Minimal 4 karakter\n- Minimal 1 huruf besar (A-Z)\n- Minimal 1 simbol (!@#$%^&* dll)")
        
        reg_username = st.text_input("Username", key="reg_user", placeholder="Min 3 characters")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("✅ Register", key="reg_btn", use_container_width=True):
            if len(reg_username) < 3:
                st.error("❌ Username minimal 3 karakter")
            else:
                is_valid, message = validate_password(reg_password)
                if not is_valid:
                    st.error(message)
                elif reg_password != reg_confirm_password:
                    st.error("❌ Password tidak sama!")
                else:
                    hashed_pw = hash_password(reg_password)
                    conn = get_connection()
                    try:
                        with conn.cursor() as cursor:
                            cursor.execute(
                                "INSERT INTO users (username, password, raw_password) VALUES (%s, %s, %s)", 
                                (reg_username, hashed_pw, reg_password)
                            )
                        conn.commit()
                        st.success("✅ Registrasi berhasil! Silakan login.")
                    except psycopg2.errors.UniqueViolation:  # ✅ PostgreSQL exception
                        conn.rollback()
                        st.error("❌ Username sudah terpakai.")
                    except Exception as e:
                        conn.rollback()
                        st.error(f"❌ Gagal mendaftar: {e}")
                    finally:
                        conn.close()
    
    # TAB 2: LOGIN
    with tab2:
        st.subheader("Login to Dashboard")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("🔓 Login", key="login_btn", type="primary", use_container_width=True):
            conn = get_connection()
            hashed_pw = hash_password(login_password)
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, username, is_admin FROM users WHERE username=%s AND password=%s", 
                        (login_username, hashed_pw)
                    )
                    user = cursor.fetchone()
                
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = user['username']
                    st.session_state.user_id = user['id']
                    st.session_state.is_admin = user['is_admin'] == 1
                    st.success("✅ Login berhasil!")
                    st.rerun()
                else:
                    st.error("❌ Username atau password salah")
            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                conn.close()
    
    # TAB 3: CHANGE PASSWORD
    with tab3:
        st.subheader("🔑 Change Password")
        
        change_username = st.text_input("Username", key="change_user")
        old_password = st.text_input("Old Password", type="password", key="old_pass")
        new_password = st.text_input("New Password", type="password", key="new_pass")
        confirm_new_password = st.text_input("Confirm New Password", type="password", key="confirm_new_pass")
        
        if st.button("🔄 Change Password", key="change_pass_btn", type="primary", use_container_width=True):
            if not all([change_username, old_password, new_password, confirm_new_password]):
                st.error("❌ Semua field harus diisi!")
            elif len(new_password) < 4:
                st.error("❌ Password baru minimal 4 karakter")
            elif new_password != confirm_new_password:
                st.error("❌ Password baru tidak sama!")
            else:
                is_valid, message = validate_password(new_password)
                if not is_valid:
                    st.error(message)
                else:
                    conn = get_connection()
                    old_hashed = hash_password(old_password)
                    new_hashed = hash_password(new_password)
                    
                    try:
                        with conn.cursor() as cursor:
                            cursor.execute(
                                "SELECT id FROM users WHERE username=%s AND password=%s", 
                                (change_username, old_hashed)
                            )
                            user = cursor.fetchone()
                            
                            if user:
                                cursor.execute(
                                    "UPDATE users SET password=%s, raw_password=%s WHERE username=%s", 
                                    (new_hashed, new_password, change_username)
                                )
                                conn.commit()
                                st.success("✅ Password berhasil diubah!")
                            else:
                                st.error("❌ Username atau password lama salah!")
                                
                    except Exception as e:
                        conn.rollback()
                        st.error(f"❌ Error: {e}")
                    finally:
                        conn.close()

# ------------------ Dashboard / Pages ------------------
def dashboard_page():
    st.title("🏠 Dashboard Overview")
    
    # Clear memory button
    col_title, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("🧹 Clear Memory"):
            st.session_state.csv_data = None
            st.session_state.cleaned_data = None
            st.cache_data.clear()
            gc.collect()
            st.success("✅ Memory cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # Get user statistics
    user_stats = get_user_statistics(st.session_state.user_id)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        data_count = len(st.session_state.csv_data) if st.session_state.csv_data is not None else 0
        st.metric("📊 Session Data", safe_metric_value(data_count))
    
    with col2:
        st.metric("💾 DB Transactions", safe_metric_value(user_stats['total_transactions']))
    
    with col3:
        st.metric("📈 DB Forecasts", safe_metric_value(user_stats['total_forecasts']))
    
    with col4:
        memory_usage = 0
        if st.session_state.csv_data is not None:
            memory_usage = st.session_state.csv_data.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("💾 Memory", f"{memory_usage:.2f} MB")
    
    st.markdown("---")
    
    # ============== LATEST FORECAST VISUALIZATION ==============
    # Get latest forecast from database
    latest_forecast = get_latest_forecast_summary(st.session_state.user_id)
    
    if latest_forecast:
        st.header("📊 Latest Forecast Visualization")
        
        # Display model info
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🤖 Model", latest_forecast['model_type'])
        with col2:
            st.metric("📅 Days", int(latest_forecast['forecast_days']))
        with col3:
            st.metric("📈 Avg", f"{float(latest_forecast['avg_value']):,.0f}")
        with col4:
            st.metric("💰 Total", f"{float(latest_forecast['total_value']):,.0f}")
        with col5:
            st.metric("📆 Date", latest_forecast['session_date'])
        
        st.markdown("---")
        
        # Load detailed forecast data
        session_date = latest_forecast['session_date']
        model_type = latest_forecast['model_type']
        
        latest_details = get_forecast_details(
            st.session_state.user_id,
            session_date,
            model_type
        )
        
        if latest_details is not None and len(latest_details) > 0:
            try:
                # Prepare data
                dates = pd.to_datetime(latest_details['Date']).values
                values = latest_details['Forecast Value'].values
                mean_val = float(np.mean(values))
                
                # ============== DUAL CHART LAYOUT ==============
                fig, (ax1) = plt.subplots(1, 1, figsize=(18, 6))
                fig.patch.set_facecolor('#0E1117')  # Match Streamlit dark theme
                
                # LEFT CHART: Line chart with trend
                ax1.set_facecolor('#262730')
                ax1.plot(dates, values, marker='o', linewidth=2.5, markersize=6,
                        color='#2E86DE', label='Forecast', zorder=3)
                
                # Trend line
                x_numeric = np.arange(len(values))
                z = np.polyfit(x_numeric, values, 1)
                p = np.poly1d(z)
                ax1.plot(dates, p(x_numeric), "--", color='#EE5A6F', 
                        linewidth=2, alpha=0.7, label='Trend', zorder=2)
                
                # Mean line
                ax1.axhline(y=mean_val, color='#26DE81', linestyle=':', 
                           linewidth=2, alpha=0.7, label=f'Mean: {mean_val:,.0f}', zorder=1)
                
                ax1.set_xlabel('Date', fontsize=12, fontweight='bold', color='white')
                ax1.set_ylabel('Forecast Value', fontsize=12, fontweight='bold', color='white')
                ax1.set_title(f'Latest Forecast - {model_type} Model', 
                             fontsize=14, fontweight='bold', pad=15, color='white')
                ax1.legend(loc='best', fontsize=10, framealpha=0.95, facecolor='#262730', edgecolor='white')
                ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.7, color='gray')
                ax1.tick_params(axis='x', rotation=45, labelsize=9, colors='white')
                ax1.tick_params(axis='y', labelsize=9, colors='white')
                
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as chart_error:
                st.error(f"❌ Error creating charts: {chart_error}")
                import traceback
                with st.expander("🔍 Chart Error Details"):
                    st.code(traceback.format_exc())
            
            st.markdown("---")
            
            # ============== FORECAST STATISTICS ==============
            st.subheader("📊 Forecast Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📈 Trend Analysis**")
                first_val = float(values[0])
                last_val = float(values[-1])
                change = last_val - first_val
                change_pct = (change / first_val) * 100 if first_val != 0 else 0
                
                if change > 0:
                    st.success(f"↗️ **Increasing Trend**")
                    st.write(f"Change: +{change:,.0f} ({change_pct:+.2f}%)")
                else:
                    st.error(f"↘️ **Decreasing Trend**")
                    st.write(f"Change: {change:,.0f} ({change_pct:.2f}%)")
                
                st.write(f"Start: {first_val:,.0f}")
                st.write(f"End: {last_val:,.0f}")
            
            with col2:
                st.markdown("**📊 Distribution**")
                std_dev = float(np.std(values))
                variance = float(np.var(values))
                min_val = float(np.min(values))
                max_val = float(np.max(values))
                range_val = max_val - min_val
                median_val = float(np.median(values))
                
                st.write(f"Mean: {mean_val:,.0f}")
                st.write(f"Median: {median_val:,.0f}")
                st.write(f"Std Dev: {std_dev:,.2f}")
                st.write(f"Range: {range_val:,.0f}")
            
            with col3:
                st.markdown("**📅 Time Period**")
                start_date = pd.to_datetime(dates[0])
                end_date = pd.to_datetime(dates[-1])
                
                st.write(f"Start: {start_date.strftime('%Y-%m-%d')}")
                st.write(f"End: {end_date.strftime('%Y-%m-%d')}")
                st.write(f"Duration: {len(latest_details)} days")
                st.write(f"Created: {session_date}")
            
            st.markdown("---")
            
            # ============== QUICK ACTIONS ==============
            col1, col2= st.columns(2)
            
            with col1:
                csv_data = latest_details.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Forecast CSV",
                    data=csv_data,
                    file_name=f"forecast_{model_type}_{session_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if st.button("📜 View All Forecasts", use_container_width=True):
                    st.session_state['show_forecast_history'] = True
                    st.rerun()
        
        else:
            st.warning("⚠️ No detailed forecast data found.")
            st.info("💡 Generate a new forecast to see visualizations here!")
    
    else:    
        st.info("ℹ️ No forecast history found. Use the '📈 Generate Forecast' menu to create your first forecast.")    
    
    st.markdown("---")
    
    # Check data size warning
    if st.session_state.csv_data is not None:
        check_data_size_warning(st.session_state.csv_data)
    
    # Latest upload info
    if user_stats['latest_upload']:
        st.info(f"📅 Latest Upload: {user_stats['latest_upload']}")
    
    # ============== DATA MANAGEMENT ACTIONS ==============
    
    col1 = st.columns(1)[0]
    with col1:
        if st.button("📥 Load My Data from Database", type="secondary", use_container_width=True):
            with st.spinner("Loading ALL your data from database..."):
                df = load_transactions_from_db(st.session_state.user_id)
                if df is not None and len(df) > 0:
                    st.session_state.csv_data = df
                    st.rerun()
                elif df is not None:
                    st.warning("⚠️ No data found in database.")
    
    # ========== FORECAST HISTORY SECTION ==========
    if st.session_state.get('show_forecast_history', False):
        st.markdown("---")
        st.header("📜 Forecast History")
        
        forecast_history = get_forecast_history(st.session_state.user_id, limit=999)
        
        if forecast_history is not None and len(forecast_history) > 0:
            st.info(f"ℹ️ Showing {len(forecast_history)} most recent forecast sessions")
            
            st.dataframe(forecast_history, use_container_width=True, height=350, hide_index=True)
        else:
            st.info("ℹ️ No forecast history found.")
        
        # Close button
        st.markdown("---")
        if st.button("❌ Close Forecast History", type="secondary", use_container_width=True):
            st.session_state['show_forecast_history'] = False
            st.rerun()
    
    # Current session data preview
    if st.session_state.csv_data is not None:
        st.markdown("---")
        st.subheader("💾 Current Session Data")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"ℹ️ Current Dataset: **{len(st.session_state.csv_data):,}** rows")
        with col2:
            st.info(f"📋 Columns: **{len(st.session_state.csv_data.columns)}**")
        
        with st.expander("👁️ Quick Preview (10 rows)"):
            st.dataframe(st.session_state.csv_data.head(10), use_container_width=True)
    else:
        st.markdown("---")
        st.warning("⚠️ No data in current session. Upload CSV or load from database.")

def display_data_page():
    st.title("📊 Tampilkan Data")
    st.markdown("---")
    
    # Option to load from database
    if st.session_state.csv_data is None:
        st.info("ℹ️ No data loaded. You can load your data from database.")
        if st.button("📥 Load My Data from Database", type="primary"):
            with st.spinner("Loading ALL your data..."):
                df = load_transactions_from_db(st.session_state.user_id)  # Hapus parameter limit
                if df is not None and len(df) > 0:
                    st.session_state.csv_data = df
                    st.rerun()
                elif df is not None:
                    st.warning("⚠️ No data found for your account.")
                else:
                    st.error("❌ Failed to load data.")
        return
    
    if st.session_state.csv_data is not None:
        total_rows = len(st.session_state.csv_data)
        st.info(f"ℹ️ Total data: **{total_rows:,}** baris (Your personal data)")
        
        # ✅ FIX: Limit rows per page to avoid MessageSizeError
        col1, col2 = st.columns(2)
        with col1:
            rows_per_page = st.selectbox(
                "Baris per halaman:", 
                options=[50, 100, 500, 1000, min(MAX_ROWS_PER_PAGE, total_rows)], 
                index=0
            )
        with col2:
            total_pages = (total_rows - 1) // rows_per_page + 1
            page_number = st.number_input(
                f"Halaman (1-{total_pages:,}):", 
                min_value=1, 
                max_value=total_pages, 
                value=1, 
                step=1
            )
        
        start_idx = (page_number - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        st.markdown("---")
        st.subheader(f"Data Preview - Halaman {page_number:,} dari {total_pages:,}")
        st.caption(f"Menampilkan baris {start_idx + 1:,} - {end_idx:,}")
        
        # ✅ FIX: Only display subset to avoid large data transfer
        display_df = st.session_state.csv_data.iloc[start_idx:end_idx]
        st.dataframe(display_df, use_container_width=True, height=500)
        
        csv_page = display_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Current Page (CSV)",
            data=csv_page,
            file_name=f"data_page_{page_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Tidak ada data. Upload file CSV di menu 'Add Data' atau load dari database.")

def cleaning_data_page():
    st.title("🧹 Cleaning Data")
    st.markdown("---")
    st.subheader("Advanced Data Cleaning (IQR Method)")
    st.write("Fitur ini akan membersihkan data menggunakan metode:")
    st.markdown("""
    - ✓ Convert tipe data (datetime & numeric)
    - ✓ Remove duplikasi data
    - ✓ Remove outliers (IQR method: Q1-1.5×IQR, Q3+1.5×IQR)
    - ✓ Handle missing values
    """)
    
    # ✅ FIXED: Initialize cleaning_done state
    if 'cleaning_done' not in st.session_state:
        st.session_state.cleaning_done = False
    
    # Option to load from database if no data
    if st.session_state.csv_data is None:
        st.info("ℹ️ No data loaded. Load your data first.")
        if st.button("📥 Load My Data from Database", type="primary"):
            with st.spinner("Loading ALL your data..."):
                df = load_transactions_from_db(st.session_state.user_id)
                if df is not None and len(df) > 0:
                    st.session_state.csv_data = df
                    st.session_state.cleaning_done = False  # Reset cleaning state
                    st.rerun()
        return
    
    if st.session_state.csv_data is not None:
        total_rows = len(st.session_state.csv_data)
        st.info(f"ℹ️ Data saat ini: **{total_rows:,}** baris (Your personal data)")
        
        # ✅ FIXED: Button always visible before cleaning
        if not st.session_state.cleaning_done:
            if st.button("🧹 Jalankan Advanced Cleaning", type="primary", use_container_width=True):
                with st.spinner("Membersihkan data dengan IQR method..."):
                    try:
                        cleaned_df, cleaning_log = clean_data_advanced(st.session_state.csv_data)
                        st.session_state.cleaned_data = cleaned_df
                        st.session_state.cleaning_done = True  # ✅ Set flag
                        st.rerun()  # ✅ Rerun to show results
                    except Exception as e:
                        st.error(f"❌ Error during cleaning: {str(e)}")
                        import traceback
                        with st.expander("🔍 Debug Info"):
                            st.code(traceback.format_exc())
        
        # ✅ FIXED: Show results and buttons AFTER cleaning
        if st.session_state.cleaning_done and st.session_state.cleaned_data is not None:
            # Re-calculate cleaning_log for display
            original_rows = len(st.session_state.csv_data)
            final_rows = len(st.session_state.cleaned_data)
            removed_rows = original_rows - final_rows
            
            st.success("✅ Data berhasil dibersihkan!")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Rows", f"{original_rows:,}")
            with col2:
                st.metric("Final Rows", f"{final_rows:,}")
            with col3:
                st.metric("Removed", f"{removed_rows:,}")
            
            st.markdown("---")
            
            # ✅ Action buttons - ALWAYS VISIBLE after cleaning
            col1, col2 = st.columns(2)
            
            with col1:
                csv_cleaned = st.session_state.cleaned_data.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Cleaned Data",
                    data=csv_cleaned,
                    file_name=f"cleaned_no_outliers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # ✅ FIXED: Save button now highly visible
                if st.button("💾 Save Cleaned Data to Database", type="primary", use_container_width=True):
                    with st.spinner("Saving cleaned data to database..."):
                        # First, delete old data for this user
                        conn = get_connection()
                        try:
                            with conn.cursor() as cursor:
                                cursor.execute("DELETE FROM transactions WHERE user_id = %s", (st.session_state.user_id,))
                                deleted_count = cursor.rowcount
                            conn.commit()
                            st.info(f"🗑️ Deleted {deleted_count:,} old records")
                        except Exception as e:
                            st.error(f"❌ Error deleting old data: {e}")
                            conn.rollback()
                        finally:
                            conn.close()
                        
                        # Save cleaned data
                        inserted = save_transactions_to_db(st.session_state.cleaned_data, st.session_state.user_id, batch_size=1000)
                        if inserted > 0:
                            st.success(f"✅ Saved {inserted:,} cleaned rows to database!")
                            
                            # ✅ Update session state
                            st.session_state.csv_data = st.session_state.cleaned_data
                            st.session_state.db_after_clean = inserted  # ✅ Set counter for sidebar
                            st.session_state.cleaning_done = False  # Reset for next cleaning
                            
                            # Refresh to show updated stats
                            st.rerun()
                        else:
                            st.error("❌ No data was saved to database")

            
            st.markdown("---")
            
            # Preview of cleaned data
            with st.expander("👁️ Preview Cleaned Data (First 10 rows)"):
                st.dataframe(st.session_state.cleaned_data.head(10), use_container_width=True)
            
            # Show cleaning details
            with st.expander("📊 View Detailed Cleaning Statistics"):
                st.write(f"**Original rows:** {original_rows:,}")
                st.write(f"**Final rows:** {final_rows:,}")
                st.write(f"**Total removed:** {removed_rows:,} ({(removed_rows/original_rows*100):.2f}%)")
            
            gc.collect()
    else:
        st.warning("⚠️ Tidak ada data untuk dibersihkan. Upload file CSV terlebih dahulu.")

def add_data_page():
    st.title("➕ Add Data")
    st.markdown("---")
    tab1, tab2 = st.tabs(["📁 Upload CSV File", "🎲 Generate Sample Data"])
    
    # ============== TAB 1: UPLOAD CSV ==============
    with tab1:
        st.subheader("Upload CSV File (Support Up To 1GB)")
        st.info("💡 Sistem mendukung file CSV hingga 1GB dengan chunked loading")
        st.warning(f"🔒 Data akan disimpan dengan user_id: {st.session_state.user_id} (Your data only)")
        
        uploaded_file = st.file_uploader(
            "Pilih file CSV", 
            type=['csv'],
            help="Maksimal 1GB - sistem akan otomatis menggunakan chunked loading untuk file besar"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File '{uploaded_file.name}' berhasil diupload!")
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📦 File size: {file_size_mb:.2f} MB")
            
            use_chunked = file_size_mb > 50
            chunk_size = 100000
            
            if use_chunked:
                st.warning(f"⚡ File besar terdeteksi ({file_size_mb:.2f} MB). Menggunakan chunked loading otomatis.")
                
                chunk_options = {
                    "10,000 rows": 10000,
                    "50,000 rows": 50000,
                    "100,000 rows (Recommended)": 100000,
                    "200,000 rows": 200000,
                    "500,000 rows": 500000
                }
                
                chunk_label = st.selectbox(
                    "Chunk size (baris per chunk):",
                    options=list(chunk_options.keys()),
                    index=2
                )
                
                chunk_size = chunk_options[chunk_label]
                st.caption(f"Selected: {chunk_size:,} rows per chunk")
            
            auto_save_to_db = st.checkbox(
                "💾 Otomatis simpan ke database setelah load",
                value=True,
                help="Data akan langsung disimpan ke tabel 'transactions' dengan user_id Anda"
            )
            
            if st.button("💾 Load Data", type="primary"):
                try:
                    with st.spinner("Loading data..."):
                        if use_chunked:
                            df = load_csv_chunked(uploaded_file, chunksize=chunk_size)
                        else:
                            df = pd.read_csv(uploaded_file, low_memory=False)
                        
                        if df is not None and len(df) > 0:
                            df = optimize_dataframe_memory(df)
                            st.session_state.csv_data = df
                            
                            st.success(f"✅ Data loaded! Total: {len(df):,} baris | {len(df.columns)} kolom")
                            
                            with st.expander("👁️ Preview (10 baris)"):
                                st.dataframe(df.head(10), use_container_width=True)
                            
                            # ✅ FIX: GUNAKAN FUNGSI DENGAN LOGGING
                            if auto_save_to_db:
                                with st.spinner("💾 Menyimpan ke database..."):
                                    inserted = save_transactions_to_db_with_logging(
                                        df=df,
                                        user_id=st.session_state.user_id,
                                        filename=uploaded_file.name,  # ← PENTING!
                                        batch_size=1000
                                    )
                                    if inserted > 0:
                                        st.success(f"✅ Berhasil menyimpan {inserted:,} baris ke database!")
                                        st.info("📝 Upload history telah dicatat!")
                                    else:
                                        st.warning("⚠️ Tidak ada data yang disimpan.")
                            
                            st.rerun()
                        else:
                            st.error("❌ Failed to load data or data is empty.")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("🔍 Full Error Details"):
                        st.code(traceback.format_exc())
    
    # ============== TAB 2: GENERATE SAMPLE ==============
    with tab2:
        st.subheader("Generate Sample Data")
        st.warning(f"🔒 Sample data akan disimpan dengan user_id: {st.session_state.user_id}")
        
        sample_size = st.number_input(
            "Jumlah baris:",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000,
            key="sample_size_input"
        )
        st.caption(f"Estimasi memory: ~{(sample_size * 40) / 1024 / 1024:.2f} MB")
        
        auto_save_sample = st.checkbox(
            "💾 Otomatis simpan sample ke database",
            value=True,
            help="Sample data akan disimpan ke database dengan user_id Anda"
        )
        
        if st.button("🎲 Generate", type="primary"):
            with st.spinner(f"Generating {sample_size:,} baris..."):
                try:
                    sample_df = generate_sample_data_chunked(sample_size)
                    st.session_state.csv_data = sample_df
                    st.success(f"✅ Generated {sample_size:,} baris!")
                    
                    # ✅ FIX: GUNAKAN FUNGSI DENGAN LOGGING
                    if auto_save_sample:
                        with st.spinner("Saving to database..."):
                            inserted = save_transactions_to_db_with_logging(
                                df=sample_df,
                                user_id=st.session_state.user_id,
                                filename=f"sample_data_{sample_size}_rows.csv",  # ← PENTING!
                                batch_size=1000
                            )
                            if inserted > 0:
                                st.success(f"✅ Saved {inserted:,} rows to database!")
                                st.info("📝 Upload history telah dicatat!")
                    
                    gc.collect()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("🔍 Debug Info"):
                        st.code(traceback.format_exc())

def delete_data_page():
    st.title("🗑️ Hapus Data")
    st.markdown("---")
    st.warning("⚠️ Halaman ini akan menghapus data dari MEMORY SESSION saja, bukan dari database.")
    st.info("💡 Untuk menghapus data dari database, gunakan menu terpisah di bawah.")
    
    if st.session_state.csv_data is not None:
        total_rows = len(st.session_state.csv_data)
        st.info(f"ℹ️ Total data di memory: **{total_rows:,}** baris")
        
        st.subheader("Hapus dari Memory Session")
        if st.button("🗑️ Clear Session Data", type="primary"):
            st.session_state.csv_data = None
            st.session_state.cleaned_data = None
            st.session_state.forecast_result = None
            gc.collect()
            st.success("✅ Session data cleared!")
            st.rerun()
    else:
        st.warning("⚠️ Tidak ada data di memory session.")
    
    st.markdown("---")
    st.subheader("🗑️ Hapus Data dari Database")
    st.error("⚠️ PERINGATAN: Ini akan menghapus SEMUA data transaksi Anda dari database secara PERMANEN!")
    
    confirm_delete = st.text_input(
        "Ketik 'DELETE ALL' untuk konfirmasi:",
        placeholder="DELETE ALL"
    )
    
    if st.button("🗑️ Delete All My Database Records", type="primary", disabled=(confirm_delete != "DELETE ALL")):
        if confirm_delete == "DELETE ALL":
            conn = get_connection()
            try:
                with conn.cursor() as cursor:
                    # Delete transactions
                    cursor.execute("DELETE FROM transactions WHERE user_id = %s", (st.session_state.user_id,))
                    deleted_trans = cursor.rowcount
                    
                    # Delete forecasts
                    cursor.execute("DELETE FROM forecasts WHERE user_id = %s", (st.session_state.user_id,))
                    deleted_forecasts = cursor.rowcount
                    
                conn.commit()
                st.success(f"✅ Deleted {deleted_trans} transactions and {deleted_forecasts} forecast records from database!")
            except Exception as e:
                conn.rollback()
                st.error(f"❌ Error deleting data: {e}")
            finally:
                conn.close()

def generate_forecast_page():
    st.title("📈 Generate Forecast")
    st.markdown("---")
    
    # Load data if not in session
    if st.session_state.csv_data is None:
        st.info("ℹ️ No data in session. Load your data first.")
        if st.button("📥 Load My Data from Database", type="primary"):
            with st.spinner("Loading ALL your data..."):
                df = load_transactions_from_db(st.session_state.user_id)
                if df is not None and len(df) > 0:
                    st.session_state.csv_data = df
                    st.rerun()
                elif df is not None:
                    st.warning("⚠️ No data found for your account.")
        return
    
    data_source = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.csv_data
    if data_source is None:
        st.warning("⚠️ Tidak ada data. Upload dan clean data terlebih dahulu.")
        return
    
    st.success(f"✅ Data tersedia: {len(data_source):,} baris (Your personal data)")
    st.info(f"🔒 Forecast akan disimpan dengan user_id: {st.session_state.user_id}")
    
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "Pilih Model Forecasting:",
            options=["SARIMA", "LSTM"],
            help="SARIMA: Statistical model, cocok untuk seasonal patterns\nLSTM: Deep Learning, cocok untuk complex patterns"
        )
        if model_choice == "SARIMA" and not SARIMA_AVAILABLE:
            st.error("❌ SARIMA tidak tersedia. Install: pip install statsmodels")
            return
        if model_choice == "LSTM" and not TENSORFLOW_AVAILABLE:
            st.error("❌ LSTM tidak tersedia. Install: pip install tensorflow")
            return
    
    with col2:
        forecast_days = st.number_input(
            "Jumlah Hari Forecast:",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Berapa hari ke depan yang ingin diprediksi"
        )
    
    st.markdown("---")
    with st.expander("🔧 Advanced Settings"):
        if model_choice == "SARIMA":
            col1, col2 = st.columns(2)
            with col1:
                seasonal_period = st.number_input("Seasonal Period:", min_value=1, max_value=365, value=7)
            with col2:
                max_params = st.number_input("Max Parameter Search:", min_value=1, max_value=5, value=2)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                sequence_length = st.number_input("Sequence Length:", min_value=7, max_value=90, value=30)
            with col2:
                lstm_units = st.number_input("LSTM Units:", min_value=10, max_value=200, value=50)
            with col3:
                epochs = st.number_input("Epochs:", min_value=10, max_value=200, value=100)
    
    st.subheader("📊 Data Preparation")
    date_cols = [col for col in data_source.columns if 'date' in col.lower()]
    if len(date_cols) == 0:
        st.error("❌ Data tidak memiliki kolom tanggal. Pastikan ada kolom 'order_date' atau 'invoice_date'")
        return
    
    date_col = st.selectbox("Pilih Kolom Tanggal:", options=date_cols)
    value_options = ['Final_Price', 'price', 'subtotal'] + [col for col in data_source.columns if col not in date_cols]
    value_col = st.selectbox("Pilih Kolom Nilai:", options=value_options)
    
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"Generating {model_choice} forecast for {forecast_days} days..."):
            try:
                # ✅ FIXED: Better time series preparation
                df_ts = data_source.copy()
                
                # ============== DATETIME CONVERSION ==============
                if date_col not in df_ts.columns:
                    st.error(f"❌ Kolom '{date_col}' tidak ditemukan dalam dataset")
                    with st.expander("🔍 Available Columns"):
                        st.write(list(df_ts.columns))
                    return
                
                # Show original data type
                st.info(f"📋 Original '{date_col}' type: {df_ts[date_col].dtype}")
                
                # Convert from categorical if needed
                if isinstance(df_ts[date_col].dtype, pd.CategoricalDtype):
                    st.info("🔄 Converting from categorical to string...")
                    df_ts[date_col] = df_ts[date_col].astype(str)
                
                # Show sample before conversion
                with st.expander("👁️ Sample Data Before Conversion"):
                    st.write(df_ts[[date_col, value_col]].head(5))
                
                # Convert to datetime with better error handling
                try:
                    df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
                    
                    # Count invalid dates
                    invalid_dates = df_ts[date_col].isna().sum()
                    if invalid_dates > 0:
                        st.warning(f"⚠️ Found {invalid_dates} invalid dates that will be removed")
                    
                except Exception as date_error:
                    st.error(f"❌ Error converting dates: {date_error}")
                    with st.expander("🔍 Sample Problematic Data"):
                        st.write(df_ts[date_col].head(10))
                    return
                
                # Remove NaT values
                before_drop = len(df_ts)
                df_ts = df_ts.dropna(subset=[date_col])
                after_drop = len(df_ts)
                
                if after_drop < before_drop:
                    st.warning(f"⚠️ Removed {before_drop - after_drop} rows with invalid dates")
                
                if len(df_ts) == 0:
                    st.error("❌ No valid data after date conversion")
                    return
                
                # ============== VALUE COLUMN VALIDATION ==============
                if value_col not in df_ts.columns:
                    st.error(f"❌ Kolom '{value_col}' tidak ditemukan")
                    return
                
                # Convert value column to numeric
                df_ts[value_col] = pd.to_numeric(df_ts[value_col], errors='coerce')
                
                # Remove invalid values
                before_drop_value = len(df_ts)
                df_ts = df_ts.dropna(subset=[value_col])
                after_drop_value = len(df_ts)
                
                if after_drop_value < before_drop_value:
                    st.warning(f"⚠️ Removed {before_drop_value - after_drop_value} rows with invalid values")
                
                if len(df_ts) == 0:
                    st.error(f"❌ No valid data in column '{value_col}'")
                    return
                
                # Sort and set index
                df_ts = df_ts.sort_values(date_col)
                df_ts.set_index(date_col, inplace=True)
                
                # Verify DatetimeIndex
                if not isinstance(df_ts.index, pd.DatetimeIndex):
                    st.error(f"❌ Index is not DatetimeIndex. Type: {type(df_ts.index)}")
                    return
                
                # Create daily time series
                daily_ts = df_ts[value_col].resample('D').sum().fillna(0)
                
                st.success(f"✅ Time series created: {len(daily_ts)} days")
                st.info(f"📅 Period: {daily_ts.index.min().strftime('%Y-%m-%d')} to {daily_ts.index.max().strftime('%Y-%m-%d')}")
                
                # ============== DATA SIZE VALIDATION ==============
                min_data_points = 60 if model_choice == "LSTM" else 40
                if len(daily_ts) < min_data_points:
                    st.error(f"❌ Insufficient data!")
                    st.write(f"- **Required:** {min_data_points} days minimum")
                    st.write(f"- **Available:** {len(daily_ts)} days")
                    st.info("💡 Please load more historical data or reduce sequence length")
                    return
                
                # Split data
                test_size = min(30, len(daily_ts) // 5)
                train_size = len(daily_ts) - test_size
                train_data = daily_ts[:train_size]
                test_data = daily_ts[train_size:]
                
                st.write(f"📊 **Training:** {len(train_data)} days | **Testing:** {len(test_data)} days")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # ============== MODEL TRAINING ==============
                if model_choice == "SARIMA":
                    status_text.text("🔄 SARIMA: Searching for best parameters...")
                    progress_bar.progress(0.2)
                    
                    sarima = SARIMAAutoForecaster()
                    summary, search_log = sarima.auto_fit(
                        train_data, 
                        max_p=max_params, 
                        max_q=max_params, 
                        max_P=1,
                        max_Q=1,
                        seasonal_period=seasonal_period
                    )
                    
                    status_text.text(f"✅ Best: SARIMA{summary['best_order']} x {summary['best_seasonal_order']}")
                    progress_bar.progress(0.5)
                    
                    model_summary = sarima.get_model_summary()
                    
                    status_text.text("🔄 Generating forecasts...")
                    progress_bar.progress(0.7)
                    
                    test_forecast, test_lower, test_upper = sarima.forecast(len(test_data))
                    future_forecast, future_lower, future_upper = sarima.forecast(forecast_days)
                    
                    metrics = {
                        'MAE': AdvancedMetrics.mae(test_data.values, test_forecast),
                        'RMSE': AdvancedMetrics.rmse(test_data.values, test_forecast),
                        'sMAPE': AdvancedMetrics.smape(test_data.values, test_forecast),
                        'MASE': AdvancedMetrics.mase(test_data.values, test_forecast, train_data.values, seasonality=seasonal_period)
                    }
                    
                    try:
                        diagnostics_fig = sarima.plot_diagnostics()
                    except:
                        diagnostics_fig = None
                    
                    st.session_state.forecast_result = {
                        'model': 'SARIMA',
                        'model_info': summary,
                        'model_summary': model_summary,
                        'search_log': search_log,
                        'diagnostics_fig': diagnostics_fig,
                        'train_data': train_data,
                        'test_data': test_data,
                        'test_forecast': test_forecast,
                        'test_lower': test_lower,
                        'test_upper': test_upper,
                        'future_forecast': future_forecast,
                        'future_lower': future_lower,
                        'future_upper': future_upper,
                        'future_dates': pd.date_range(start=daily_ts.index[-1] + timedelta(days=1), periods=forecast_days, freq='D'),
                        'metrics': metrics,
                        'forecast_days': forecast_days
                    }
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ SARIMA forecast completed!")
                
                else:  # LSTM
                    status_text.text(f"🔄 Training LSTM (Sequence: {sequence_length}, Units: {lstm_units})...")
                    progress_bar.progress(0.3)
                    
                    lstm = LSTMForecaster(sequence_length=sequence_length, lstm_units=lstm_units)
                    history = lstm.fit(train_data, epochs=epochs)
                    
                    status_text.text("🔄 Generating forecasts...")
                    progress_bar.progress(0.6)
                    
                    test_forecast = lstm.forecast(train_data, len(test_data))
                    future_forecast = lstm.forecast(train_data, forecast_days)
                    
                    metrics = {
                        'MAE': AdvancedMetrics.mae(test_data.values, test_forecast),
                        'RMSE': AdvancedMetrics.rmse(test_data.values, test_forecast),
                        'sMAPE': AdvancedMetrics.smape(test_data.values, test_forecast),
                        'MASE': AdvancedMetrics.mase(test_data.values, test_forecast, train_data.values, seasonality=7)
                    }
                    
                    st.session_state.forecast_result = {
                        'model': 'LSTM',
                        'model_info': {
                            'sequence_length': sequence_length,
                            'lstm_units': lstm_units,
                            'epochs': epochs
                        },
                        'train_data': train_data,
                        'test_data': test_data,
                        'test_forecast': test_forecast,
                        'future_forecast': future_forecast,
                        'future_dates': pd.date_range(start=daily_ts.index[-1] + timedelta(days=1), periods=forecast_days, freq='D'),
                        'metrics': metrics,
                        'forecast_days': forecast_days
                    }
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ LSTM forecast completed!")
                
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"🎉 {model_choice} forecast berhasil!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during forecasting: {str(e)}")
                import traceback
                with st.expander("🔍 Full Error Traceback"):
                    st.code(traceback.format_exc())
                    
                # Show helpful debugging info
                with st.expander("🔧 Debugging Information"):
                    st.write("**Data Source Info:**")
                    st.write(f"- Total rows: {len(data_source):,}")
                    st.write(f"- Columns: {list(data_source.columns)}")
                    if date_col in data_source.columns:
                        st.write(f"- Date column type: {data_source[date_col].dtype}")
                        st.write(f"- Date sample: {data_source[date_col].head(3).tolist()}")
    
    # ============== DISPLAY RESULTS ==============
    if st.session_state.forecast_result:
        st.markdown("---")
        st.header("📊 Hasil Forecast")
        result = st.session_state.forecast_result
        
        st.subheader(f"🤖 Model: {result['model']}")
        
        if result['model'] == 'SARIMA':
            st.info(f"✅ SARIMA {result['model_info']['best_order']} x {result['model_info']['best_seasonal_order']}")
        else:
            st.info(f"✅ LSTM (Seq: {result['model_info']['sequence_length']}, Units: {result['model_info']['lstm_units']})")
        
        st.markdown("---")
        
        # ============== PLOTTING WITH DARK THEME ==============
        try:
            fig, ax = plt.subplots(1, 1, figsize=(16, 7))
            
            # ✅ FIX: Dark theme for Streamlit
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#262730')
            
            # Plot historical data (last 90 days)
            recent_train = result['train_data'].tail(90)
            
            # Sample if too many points
            MAX_PLOT_POINTS = 2000
            if len(recent_train) > MAX_PLOT_POINTS:
                sample_indices = np.linspace(0, len(recent_train)-1, MAX_PLOT_POINTS, dtype=int)
                recent_train_sampled = recent_train.iloc[sample_indices]
                ax.plot(recent_train_sampled.index, recent_train_sampled.values, 
                       label='Historical', linewidth=2.5, alpha=0.8, color='#3498db')
                st.caption(f"📊 Displaying {MAX_PLOT_POINTS} sampled points")
            else:
                ax.plot(recent_train.index, recent_train.values, 
                       label='Historical', linewidth=2.5, alpha=0.8, color='#3498db')
            
            # Plot test actual
            ax.plot(result['test_data'].index, result['test_data'].values, 
                   label='Actual (Test)', linewidth=2.5, marker='o', markersize=5, 
                   color='#2ecc71', zorder=5)
            
            # Plot test forecast
            ax.plot(result['test_data'].index, result['test_forecast'], 
                   label='Forecast (Test)', linewidth=2, linestyle='--', marker='s', 
                   markersize=4, color='#e74c3c', zorder=4)
            
            # Plot future forecast with connection
            future_dates_conn = [result['train_data'].index[-1]] + list(result['future_dates'])
            future_values_conn = [result['train_data'].iloc[-1]] + list(result['future_forecast'])
            ax.plot(future_dates_conn, future_values_conn, 
                   label=f'Future ({result["forecast_days"]}d)', 
                   linewidth=2.5, linestyle='--', marker='o', markersize=5, 
                   color='#f39c12', zorder=3)
            
            # Styling
            ax.set_title(f'{result["model"]} Forecast Results', 
                        fontsize=15, fontweight='bold', color='white', pad=20)
            ax.set_xlabel('Date', fontsize=12, fontweight='bold', color='white')
            ax.set_ylabel('Value', fontsize=12, fontweight='bold', color='white')
            ax.legend(loc='best', fontsize=11, framealpha=0.9, 
                     facecolor='#262730', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
            ax.tick_params(axis='x', rotation=45, colors='white', labelsize=10)
            ax.tick_params(axis='y', colors='white', labelsize=10)
            
            # Add "Today" marker
            ax.axvline(x=result['train_data'].index[-1], color='yellow', 
                      linestyle=':', linewidth=2, alpha=0.7)
            ax.text(result['train_data'].index[-1], ax.get_ylim()[1]*0.95, 
                   'Today', ha='center', fontsize=11, color='yellow', 
                   fontweight='bold', bbox=dict(boxstyle='round', 
                   facecolor='#262730', alpha=0.8, edgecolor='yellow'))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # ✅ CRITICAL: Close figure to free memory
            plt.close(fig)
            
        except Exception as plot_error:
            st.error(f"❌ Plotting error: {plot_error}")
            import traceback
            with st.expander("🔍 Plot Error Details"):
                st.code(traceback.format_exc())
        
        # ============== METRICS & DOWNLOADS ==============
        st.markdown("---")
        st.subheader("📈 Forecast Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Period", f"{result['forecast_days']} days")
        with col2:
            avg_fc = np.mean(result['future_forecast'])
            st.metric("Average", f"{avg_fc:,.0f}")
        with col3:
            trend = "↗️ Up" if result['future_forecast'][-1] > result['future_forecast'][0] else "↘️ Down"
            st.metric("Trend", trend)
        with col4:
            total_fc = np.sum(result['future_forecast'])
            st.metric("Total", f"{total_fc:,.0f}")
        
        st.markdown("---")
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{result['metrics']['MAE']:,.2f}")
        with col2:
            st.metric("RMSE", f"{result['metrics']['RMSE']:,.2f}")
        with col3:
            st.metric("sMAPE", f"{result['metrics']['sMAPE']:.2f}%")
        with col4:
            st.metric("MASE", f"{result['metrics']['MASE']:.2f}")
        
        st.markdown("---")
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': result['future_dates'],
            'Forecast': result['future_forecast']
        })
        
        if result['model'] == 'SARIMA':
            forecast_df['Lower_CI'] = result['future_lower']
            forecast_df['Upper_CI'] = result['future_upper']
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_fc = forecast_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_fc,
                file_name=f"forecast_{result['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            metrics_json = {
                'model': result['model'],
                'model_info': result['model_info'],
                'metrics': result['metrics'],
                'forecast_days': result['forecast_days']
            }
            st.download_button(
                label="⬇️ Download Metrics",
                data=str(metrics_json),
                file_name=f"metrics_{result['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            if st.button("💾 Save to DB", use_container_width=True, type="primary"):
                with st.spinner("Saving to database..."):
                    inserted = save_forecasts_to_db(
                        st.session_state.user_id,
                        result['model'], 
                        result['future_dates'], 
                        result['future_forecast']
                    )
                    if inserted > 0:
                        st.success(f"✅ Saved {inserted:,} records!")
                    else:
                        st.error("❌ Failed to save")
        
        # Forecast table
        with st.expander("📋 View Forecast Table"):
            st.dataframe(forecast_df, use_container_width=True, height=400)


def admin_dashboard_page():
    """
    Admin dashboard page - FIXED FOR LOADING ISSUE
    """
    st.title("🔐 Admin Dashboard")
    st.markdown("---")
    
    # ✅ FIX: Add loading indicator
    with st.spinner("Loading admin dashboard..."):
        try:
            # Load data with timeout protection
            users_df = get_all_users_admin()
            uploads_df = get_latest_uploads_admin()
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("👥 Total Users", len(users_df))
            with col2:
                total_trans = int(users_df['total_transactions'].sum()) if len(users_df) > 0 else 0
                st.metric("💾 Total Transactions", f"{total_trans:,}")
            with col3:
                total_forecasts = int(users_df['total_forecasts'].sum()) if len(users_df) > 0 else 0
                st.metric("📈 Total Forecasts", total_forecasts)
            with col4:
                st.metric("📤 Total Uploads", len(uploads_df))
            
            st.markdown("---")
            
            # Tabs
            tab1, tab2 = st.tabs(["👥 User Management", "📤 Upload History"])
            
            with tab1:
                st.subheader("👥 Registered Users")
                
                if len(users_df) > 0:
                    display_users = users_df.copy()
                    display_users['created_at'] = pd.to_datetime(display_users['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    display_users = display_users.rename(columns={
                        'id': 'ID',
                        'username': 'Username',
                        'raw_password': 'Password',
                        'created_at': 'Registered',
                        'total_transactions': 'Transactions',
                        'total_forecasts': 'Forecasts'
                    })
                    
                    st.dataframe(
                        display_users[['ID', 'Username', 'Password', 'Registered', 'Transactions', 'Forecasts']], 
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download button
                    csv_users = display_users.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download User List (CSV)",
                        data=csv_users,
                        file_name=f"users_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    st.markdown("---")
                    st.subheader("🔧 User Actions")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔑 Reset User Password**")
                        user_to_reset = st.selectbox(
                            "Select User:",
                            options=users_df['username'].tolist(),
                            key="reset_user_select"
                        )
                        new_password = st.text_input(
                            "New Password:",
                            type="password",
                            key="new_password_input",
                            placeholder="Min 4 chars, 1 uppercase, 1 symbol"
                        )
                        
                        if st.button("🔄 Reset Password", key="reset_pw_btn"):
                            if new_password:
                                is_valid, message = validate_password(new_password)
                                if not is_valid:
                                    st.error(message)
                                else:
                                    user_id = users_df[users_df['username'] == user_to_reset]['id'].iloc[0]
                                    if reset_user_password_admin(user_id, new_password):
                                        st.success(f"✅ Password for '{user_to_reset}' has been reset!")
                                        st.rerun()
                            else:
                                st.error("❌ Please enter new password")
                    
                    with col2:
                        st.markdown("**🗑️ Delete User**")
                        user_to_delete = st.selectbox(
                            "Select User:",
                            options=users_df['username'].tolist(),
                            key="delete_user_select"
                        )
                        confirm_delete = st.text_input(
                            "Type 'DELETE' to confirm:",
                            key="confirm_delete_input",
                            placeholder="DELETE"
                        )
                        
                        if st.button("🗑️ Delete User", key="delete_user_btn", type="primary"):
                            if confirm_delete == "DELETE":
                                user_id = users_df[users_df['username'] == user_to_delete]['id'].iloc[0]
                                if delete_user_admin(user_id):
                                    st.success(f"✅ User '{user_to_delete}' has been deleted!")
                                    st.rerun()
                            else:
                                st.error("❌ Please type 'DELETE' to confirm")
                else:
                    st.info("ℹ️ No users registered yet")
            
            with tab2:
                st.subheader("📤 Recent CSV Uploads")
                
                if len(uploads_df) > 0:
                    display_uploads = uploads_df.copy()
                    display_uploads['upload_date'] = pd.to_datetime(display_uploads['upload_date']).dt.strftime('%Y-%m-%d %H:%M')
                    display_uploads['file_size_mb'] = display_uploads['file_size_mb'].apply(lambda x: f"{float(x):.2f} MB" if pd.notna(x) else "N/A")
                    display_uploads['total_rows'] = display_uploads['total_rows'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "0")
                    
                    def format_status(status):
                        if status == 'success':
                            return "✅ Success"
                        elif status == 'failed':
                            return "❌ Failed"
                        else:
                            return "⚠️ Partial"
                        
                    display_uploads['status'] = display_uploads['status'].apply(format_status)
                    
                    display_uploads = display_uploads.rename(columns={
                        'username': 'User',
                        'filename': 'File Name',
                        'file_size_mb': 'Size',
                        'total_rows': 'Rows',
                        'upload_date': 'Upload Date',
                        'status': 'Status'
                    })
                    
                    st.dataframe(
                        display_uploads[['User', 'File Name', 'Size', 'Rows', 'Upload Date', 'Status']], use_container_width=True, height=500)
                    
                    # Download button
                    csv_uploads = display_uploads.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Upload History (CSV)",
                        data=csv_uploads,
                        file_name=f"upload_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv")
                    
                    # Show errors if any
                    failed_uploads = uploads_df[uploads_df['status'] == 'failed']
                    if len(failed_uploads) > 0:
                        with st.expander(f"⚠️ Failed Uploads ({len(failed_uploads)})"):
                            for idx, row in failed_uploads.iterrows():
                                st.error(f"**{row['username']}** - {row['filename']}")
                                if row.get('error_message'):
                                    st.code(row['error_message'])
                else:
                    st.info("ℹ️ No upload history found")
                    st.caption("💡 Upload history will appear here after users upload CSV files")
                    
        except Exception as e:
            st.error(f"❌ Error loading admin dashboard: {e}")
            import traceback
            with st.expander("🔍 Full Error Details"):
                st.code(traceback.format_exc())

def admin_management_page():
    st.title("🔐 Admin Management")
    st.markdown("---")
    
    users_df = get_all_users_admin()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Total Users", len(users_df))
    with col2:
        total_trans = users_df['total_transactions'].sum() if len(users_df) > 0 else 0
        st.metric("💾 Total Transactions", f"{int(total_trans):,}")
    with col3:
        total_forecasts = users_df['total_forecasts'].sum() if len(users_df) > 0 else 0
        st.metric("📈 Total Forecasts", int(total_forecasts))
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["👥 User List", "🔧 User Actions"])
    
    with tab1:
        st.subheader("Registered Users")
        if len(users_df) > 0:
            display_users = users_df.copy()
            display_users['created_at'] = pd.to_datetime(display_users['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            display_users = display_users.rename(columns={
                'id': 'ID', 'username': 'Username', 'raw_password': 'Password',
                'created_at': 'Registered', 'total_transactions': 'Transactions',
                'total_forecasts': 'Forecasts'
            })
            st.dataframe(display_users[['ID', 'Username', 'Password', 'Registered', 'Transactions', 'Forecasts']],
                        use_container_width=True, height=400)
        else:
            st.info("ℹ️ No users registered yet")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔑 Reset User Password**")
            if len(users_df) > 0:
                user_to_reset = st.selectbox("Select User:", options=users_df['username'].tolist(), key="reset_user")
                new_password = st.text_input("New Password:", type="password", key="new_password")
                if st.button("🔄 Reset Password", key="reset_pw_btn"):
                    if new_password:
                        is_valid, message = validate_password(new_password)
                        if not is_valid:
                            st.error(message)
                        else:
                            user_id = users_df[users_df['username'] == user_to_reset]['id'].iloc[0]
                            if reset_user_password_admin(user_id, new_password):
                                st.success(f"✅ Password reset for '{user_to_reset}'!")
                                st.rerun()
        
        with col2:
            st.markdown("**🗑️ Delete User**")
            if len(users_df) > 0:
                user_to_delete = st.selectbox("Select User:", options=users_df['username'].tolist(), key="delete_user")
                confirm_delete = st.text_input("Type 'DELETE' to confirm:", key="confirm_delete")
                if st.button("🗑️ Delete User", key="delete_user_btn", type="primary"):
                    if confirm_delete == "DELETE":
                        user_id = users_df[users_df['username'] == user_to_delete]['id'].iloc[0]
                        if delete_user_admin(user_id):
                            st.success(f"✅ User '{user_to_delete}' deleted!")
                            st.rerun()

# ------------------ MAIN ------------------
def main():
    """
    Main application entry point - FIXED FOR LOADING ISSUE
    """
    
    # ============== INITIALIZATION ==============
    if 'schema_checked' not in st.session_state:
        check_and_update_schema()
        st.session_state.schema_checked = True
    
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    
    if 'admin_view_mode' not in st.session_state:
        st.session_state.admin_view_mode = "User Mode"
    
    # ✅ CRITICAL FIX: Prevent infinite rerun
    if 'page_loaded' not in st.session_state:
        st.session_state.page_loaded = False
    
    # ============== LOGIN CHECK ==============
    if not st.session_state.logged_in:
        login_page()
        return
    
    # ============== SIDEBAR ==============
    with st.sidebar:
        st.title("PT. XYZ")
        st.markdown("**Dashboard System**")
        st.markdown("---")
        st.write(f"👤 **{st.session_state.username}**")
        
        # ============== ADMIN vs USER SECTION ==============
        if st.session_state.is_admin:
            st.success("🔐 **ADMIN ACCESS**")
            st.markdown("---")
            
            # ✅ CRITICAL FIX: Use index instead of current value to prevent rerun
            current_index = 0 if st.session_state.admin_view_mode == "User Mode" else 1
            
            view_mode = st.radio(
                "Select Mode:",
                options=["User Mode", "Admin Management"],
                index=current_index,
                help="Switch between user features and admin management"
            )
            
            # ✅ CRITICAL FIX: Only update if changed
            if view_mode != st.session_state.admin_view_mode:
                st.session_state.admin_view_mode = view_mode
                st.rerun()  # Explicit rerun only when changed
            
            st.markdown("---")
            
            # Statistics based on mode
            if view_mode == "User Mode":
                st.info("💡 Using dashboard as regular user")
                
                # ✅ FIX: Use try-except to prevent crash
                try:
                    user_stats = get_user_statistics(st.session_state.user_id)
                    st.subheader("📊 My Statistics")
                    st.metric("DB Transactions", safe_metric_value(user_stats['total_transactions']))
                    st.metric("DB Forecasts", safe_metric_value(user_stats['total_forecasts']))
                except Exception as e:
                    st.error(f"Error loading stats: {e}")
                
            else:  # Admin Management Mode
                st.info("💡 Managing all users")
        
        else:
            # Regular user
            st.caption(f"User ID: {st.session_state.user_id}")
            
            # ✅ FIX: Use try-except to prevent crash
            try:
                user_stats = get_user_statistics(st.session_state.user_id)
                st.markdown("---")
                st.subheader("📊 Your Statistics")
                st.metric("DB Transactions", safe_metric_value(user_stats['total_transactions']))
                st.metric("DB Forecasts", safe_metric_value(user_stats['total_forecasts']))
            except Exception as e:
                st.error(f"Error loading stats: {e}")
        
        st.markdown("---")
        
        # ============== MENU ==============
        menu = None
        
        if not st.session_state.is_admin or st.session_state.admin_view_mode == "User Mode":
            # ✅ FIX: Use key to prevent state conflicts
            menu = st.radio(
                "Navigation",
                options=[
                    "🏠 Dashboard",
                    "📊 Tampilkan Data",
                    "🧹 Cleaning Data",
                    "➕ Add Data",
                    "🗑️ Hapus Data",
                    "📈 Generate Forecast"
                ],
                key="main_menu_radio",
                label_visibility="visible"
            )
            st.markdown("---")
        
        # ============== LOGOUT ==============
        if st.button("🚪 Logout", use_container_width=True, type="primary", key="logout_btn"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            gc.collect()
            st.rerun()
    
    # ============== ROUTING ==============
    # ✅ CRITICAL FIX: Wrap in try-except to catch any rendering errors
    try:
        if st.session_state.is_admin and st.session_state.admin_view_mode == "Admin Management":
            # Admin Management Page
            admin_dashboard_page()
            
        elif menu is not None:
            # User Pages
            if menu == "🏠 Dashboard":
                dashboard_page()
            elif menu == "📊 Tampilkan Data":
                display_data_page()
            elif menu == "🧹 Cleaning Data":
                cleaning_data_page()
            elif menu == "➕ Add Data":
                add_data_page()
            elif menu == "🗑️ Hapus Data":
                delete_data_page()
            elif menu == "📈 Generate Forecast":
                generate_forecast_page()
        else:
            st.warning("⚠️ Please select a menu option from the sidebar")
            
    except Exception as e:
        st.error(f"❌ Error rendering page: {e}")
        import traceback
        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc())
        
        # Show safe recovery option
        if st.button("🔄 Reload Application", type="primary"):
            st.session_state.clear()
            st.rerun()


# ============== APPLICATION ENTRY POINT ==============
if __name__ == "__main__":
    main()
