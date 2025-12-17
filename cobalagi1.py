import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import gc
import warnings
import matplotlib.pyplot as plt
import hashlib
import pymysql.cursors

warnings.filterwarnings('ignore')

# ============== PAGE CONFIG (ADD THIS AT THE TOP) ==============
st.set_page_config(
    page_title="PT. XYZ Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CONSTANTS ==============
MAX_ROWS_IN_MEMORY = 100000  # Max rows to keep in session state
MAX_ROWS_PER_PAGE = 5000     # Max rows per dataframe page
MAX_PLOT_POINTS = 2000       # Max points for plotting

# ------------------ DATABASE CONNECTION ------------------
def get_connection():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='pt_xyz',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )
    return connection

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
    """Optimize dataframe memory usage"""
    df_optimized = df.copy()
    
    # Optimize numeric columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
    
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Convert object columns to category if beneficial
    for col in df_optimized.select_dtypes(include=['object']).columns:
        num_unique = df_optimized[col].nunique()
        num_total = len(df_optimized[col])
        if num_unique / num_total < 0.5:
            df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized

# ------------------ DATABASE SCHEMA CHECKER ------------------
def check_and_update_schema():
    """Check and update database schema to include user_id columns"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Check transactions table
            cursor.execute("SHOW COLUMNS FROM transactions LIKE 'user_id'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE transactions ADD COLUMN user_id INT NOT NULL DEFAULT 0 AFTER id")
                cursor.execute("ALTER TABLE transactions ADD INDEX idx_user_id (user_id)")
                st.info("✅ Added user_id column to transactions table")
            
            # Check forecasts table
            cursor.execute("SHOW COLUMNS FROM forecasts LIKE 'user_id'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE forecasts ADD COLUMN user_id INT NOT NULL DEFAULT 0 AFTER id")
                cursor.execute("ALTER TABLE forecasts ADD INDEX idx_user_id (user_id)")
                st.info("✅ Added user_id column to forecasts table")
            
        conn.commit()
    except Exception as e:
        st.warning(f"Schema check: {e}")
    finally:
        conn.close()

# ------------------ Utilities ------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def save_transactions_to_db(df: pd.DataFrame, user_id: int, batch_size=1000):
    """
    Save transactions to database with batch processing
    Optimized for large datasets
    """
    if df is None or len(df) == 0 or user_id is None:
        st.error("❌ Data kosong atau user_id tidak valid")
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
    
    # Build SQL
    placeholders = ", ".join(["%s"] * len(db_cols))
    cols_sql = ", ".join([f"`{c}`" for c in db_cols])
    insert_sql = f"INSERT INTO transactions ({cols_sql}) VALUES ({placeholders})"
    
    conn = None
    inserted = 0
    skipped = 0
    
    try:
        conn = get_connection()
        
        # Process in batches
        total_batches = (len(df) - 1) // batch_size + 1
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            values = []
            
            for _, row in batch_df.iterrows():
                row_vals = [user_id]
                skip_row = False
                
                for col in available_cols:
                    v = row[col]
                    
                    # Check required fields
                    if col in required_cols and pd.isna(v):
                        skipped += 1
                        skip_row = True
                        break
                    
                    # Convert values
                    if pd.isna(v):
                        row_vals.append(None)
                    elif isinstance(v, (pd.Timestamp, datetime)):
                        row_vals.append(v.strftime("%Y-%m-%d %H:%M:%S"))
                    elif isinstance(v, (int, float)):
                        row_vals.append(float(v) if col in ['price', 'Final_Price', 'subtotal', 'discount'] else int(v))
                    else:
                        row_vals.append(str(v))
                
                if not skip_row:
                    values.append(tuple(row_vals))
            
            # Insert batch
            if values:
                with conn.cursor() as cur:
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
            st.warning(f"⚠️ {skipped:,} baris dilewati (order_date atau Final_Price kosong)")
        
        st.success(f"✅ Berhasil menyimpan {inserted:,} transaksi ke database!")
        
        return inserted
        
    except pymysql.err.IntegrityError as e:
        if conn:
            conn.rollback()
        st.error(f"❌ Data duplikat atau constraint error: {e}")
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
    Test database connection and show database info
    """
    conn = None
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            # Test connection
            cursor.execute("SELECT 1")
            
            # Get database info
            cursor.execute("SELECT DATABASE()")
            db_name = cursor.fetchone()
            
            # Get table info
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            # Get transactions table structure
            cursor.execute("DESCRIBE transactions")
            structure = cursor.fetchall()
        
        st.success("✅ Database connection successful!")
        st.info(f"📊 Database: {db_name}")
        st.write("📋 Available tables:", [list(t.values())[0] for t in tables])
        
        with st.expander("🔍 Transactions Table Structure"):
            st.dataframe(pd.DataFrame(structure))
        
        return True
        
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.info("💡 Check: MySQL running, database created, credentials correct")
        return False
        
    finally:
        if conn:
            conn.close()

def load_transactions_from_db(user_id: int, limit=MAX_ROWS_IN_MEMORY, offset=0, date_from=None, date_to=None):
    """
    Load transactions with optimized query and better error handling
    
    Args:
        user_id: User ID to filter
        limit: Maximum rows to load
        offset: Starting position (for pagination)
        date_from: Filter from date (optional)
        date_to: Filter to date (optional)
    """
    if user_id is None:
        st.error("❌ User ID tidak ditemukan. Silakan login ulang.")
        return None
    
    conn = None
    try:
        conn = get_connection()
        
        # Build query with optional date filters
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
                Cat2,
                qty,
                price,
                bundle_price,
                subtotal,
                discount,
                shipping_fee,
                Final_Price,
                bulan,
                year,
                month,
                created_at
            FROM transactions 
            WHERE user_id = %s
        """
        
        params = [user_id]
        
        # Add date filters if provided
        if date_from:
            base_query += " AND order_date >= %s"
            params.append(date_from)
        
        if date_to:
            base_query += " AND order_date <= %s"
            params.append(date_to)
        
        # Add ordering and pagination
        base_query += " ORDER BY order_date DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        # Get total count for info
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
            
            # Load data
            cursor.execute(base_query, params)
            results = cursor.fetchall()
        
        if len(results) == 0:
            st.warning(f"⚠️ Tidak ada data transaksi ditemukan untuk user_id: {user_id}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Convert date columns
        date_columns = ['order_date', 'invoice_date', 'created_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Optimize memory
        df = optimize_dataframe_memory(df)
        
        # Show info
        if offset == 0:
            st.success(f"✅ Loaded {len(df):,} rows from database (Total in DB: {total_rows:,})")
            if len(df) < total_rows:
                st.info(f"📊 Showing most recent {len(df):,} of {total_rows:,} total transactions")
        
        return df
        
    except pymysql.err.OperationalError as e:
        st.error(f"❌ Database connection error: {e}")
        st.info("💡 Pastikan MySQL server berjalan dan kredensial database benar")
        return None
        
    except pymysql.err.ProgrammingError as e:
        st.error(f"❌ SQL query error: {e}")
        st.info("💡 Periksa struktur tabel 'transactions' di database")
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
    """Save forecasts to database with user_id"""
    if user_id is None:
        st.error("User tidak ditemukan. Login terlebih dahulu untuk menyimpan forecast.")
        return 0
    
    insert_sql = "INSERT INTO forecasts (user_id, username, model_type, forecast_date, forecast_value) VALUES (%s, %s, %s, %s, %s)"
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
        st.error(f"Failed to save forecasts to DB: {e}")
    finally:
        conn.close()
    return inserted

def get_forecast_history(user_id: int, limit=10):
    """Get forecast history for a specific user"""
    if user_id is None:
        return None
    
    conn = get_connection()
    try:
        query = """
            SELECT model_type, forecast_date, forecast_value, created_at 
            FROM forecasts 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(user_id, limit))
        return df
    except Exception as e:
        st.error(f"❌ Failed to load forecast history: {e}")
        return None
    finally:
        conn.close()

def get_user_statistics(user_id: int):
    """Get statistics for a specific user"""
    if user_id is None:
        return {'total_transactions': 0, 'total_forecasts': 0, 'latest_upload': None}
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Total transactions
            cursor.execute("SELECT COUNT(*) as count FROM transactions WHERE user_id = %s", (user_id,))
            total_transactions = cursor.fetchone()['count']
            
            # Total forecasts
            cursor.execute("SELECT COUNT(DISTINCT created_at) as count FROM forecasts WHERE user_id = %s", (user_id,))
            total_forecasts = cursor.fetchone()['count']
            
            # Latest upload
            cursor.execute("SELECT MAX(created_at) as latest FROM transactions WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            latest_upload = result['latest'] if result else None
        
        return {
            'total_transactions': total_transactions,
            'total_forecasts': total_forecasts,
            'latest_upload': latest_upload
        }
    except Exception as e:
        st.error(f"❌ Failed to get user statistics: {e}")
        return {'total_transactions': 0, 'total_forecasts': 0, 'latest_upload': None}
    finally:
        conn.close()

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
        train_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
        X, y = self._create_sequences(train_scaled, self.sequence_length)
        if len(X) < 50:
            raise Exception(f"Not enough data for LSTM. Need at least {self.sequence_length + 50} points")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(self.lstm_units//2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        from tensorflow.keras.callbacks import EarlyStopping
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
def login_page():
    st.title("PT. XYZ")
    tab1, tab2, tab3 = st.tabs(["📝 Register", "🔐 Login", "🔑 Change Password"])
    
    # TAB 1: REGISTER
    with tab1:
        st.subheader("Register New Account")
        reg_username = st.text_input("Username", key="reg_user", placeholder="Min 3 characters")
        reg_password = st.text_input("Password", type="password", key="reg_pass", placeholder="Min 4 characters")
        reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm", placeholder="Re-enter password")
        
        if st.button("✅ Register", key="reg_btn", use_container_width=True):
            if len(reg_username) < 3:
                st.error("❌ Username minimal 3 karakter")
            elif len(reg_password) < 4:
                st.error("❌ Password minimal 4 karakter")
            elif reg_password != reg_confirm_password:
                st.error("❌ Password dan Confirm Password tidak sama!")
            else:
                hashed_pw = hash_password(reg_password)
                conn = get_connection()
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", 
                                     (reg_username, hashed_pw))
                    conn.commit()
                    st.success("✅ Registrasi berhasil! Silakan login.")
                except pymysql.err.IntegrityError:
                    st.error("❌ Username sudah terpakai. Gunakan username lain.")
                except Exception as e:
                    st.error(f"❌ Gagal mendaftar: {e}")
                finally:
                    conn.close()
    
    # TAB 2: LOGIN
    with tab2:
        st.subheader("Login to Dashboard")
        login_username = st.text_input("Username", key="login_user", placeholder="Enter username")
        login_password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
        
        if st.button("🔓 Login", key="login_btn", type="primary", use_container_width=True):
            conn = get_connection()
            hashed_pw = hash_password(login_password)
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id, username FROM users WHERE username=%s AND password=%s", 
                                 (login_username, hashed_pw))
                    user = cursor.fetchone()
                
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = user['username']
                    st.session_state.user_id = user['id']
                    st.success("✅ Login berhasil!")
                    st.rerun()
                else:
                    st.error("❌ Username atau password salah")
            except Exception as e:
                st.error(f"❌ Error saat login: {e}")
            finally:
                conn.close()
    
    # TAB 3: CHANGE PASSWORD
    with tab3:
        st.subheader("🔑 Change Password")
        st.info("💡 Masukkan username dan password lama Anda, kemudian masukkan password baru.")
        
        change_username = st.text_input("Username", key="change_user", placeholder="Enter your username")
        old_password = st.text_input("Old Password", type="password", key="old_pass", placeholder="Enter current password")
        new_password = st.text_input("New Password", type="password", key="new_pass", placeholder="Min 4 characters")
        confirm_new_password = st.text_input("Confirm New Password", type="password", key="confirm_new_pass", placeholder="Re-enter new password")
        
        if st.button("🔄 Change Password", key="change_pass_btn", type="primary", use_container_width=True):
            if not change_username or not old_password or not new_password or not confirm_new_password:
                st.error("❌ Semua field harus diisi!")
            elif len(new_password) < 4:
                st.error("❌ Password baru minimal 4 karakter")
            elif new_password != confirm_new_password:
                st.error("❌ Password baru dan konfirmasi tidak sama!")
            elif old_password == new_password:
                st.warning("⚠️ Password baru tidak boleh sama dengan password lama!")
            else:
                conn = get_connection()
                old_hashed = hash_password(old_password)
                new_hashed = hash_password(new_password)
                
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT id FROM users WHERE username=%s AND password=%s", 
                                     (change_username, old_hashed))
                        user = cursor.fetchone()
                        
                        if user:
                            cursor.execute("UPDATE users SET password=%s WHERE username=%s", 
                                         (new_hashed, change_username))
                            conn.commit()
                            st.success("✅ Password berhasil diubah! Silakan login dengan password baru.")
                        else:
                            st.error("❌ Username atau password lama salah!")
                            
                except Exception as e:
                    conn.rollback()
                    st.error(f"❌ Error saat mengubah password: {e}")
                finally:
                    conn.close()

# ------------------ Dashboard / Pages ------------------
def dashboard_page():
    st.title("🏠 Dashboard Overview")
    
    # ✅ FIX: Add clear cache button
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
    
    # Get user statistics from database
    user_stats = get_user_statistics(st.session_state.user_id)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # ✅ FIX: Send numeric value directly (not f-string)
        data_count = len(st.session_state.csv_data) if st.session_state.csv_data is not None else 0
        st.metric("📊 Current Session Data", safe_metric_value(data_count))
    
    with col2:
        # ✅ FIX: Send numeric value directly
        st.metric("💾 Total Transactions", safe_metric_value(user_stats['total_transactions']))
    
    with col3:
        # ✅ FIX: Send numeric value directly
        st.metric("📈 Total Forecasts", safe_metric_value(user_stats['total_forecasts']))
    
    with col4:
        memory_usage = 0
        if st.session_state.csv_data is not None:
            memory_usage = st.session_state.csv_data.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("💾 Memory", f"{memory_usage:.2f} MB")
    
    st.markdown("---")
    
    # ✅ FIX: Check data size
    if st.session_state.csv_data is not None:
        check_data_size_warning(st.session_state.csv_data)
    
    # Display latest upload info
    if user_stats['latest_upload']:
        st.info(f"📅 Latest Upload: {user_stats['latest_upload']}")
    
    # Load data from database button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Load My Data from Database", type="primary", use_container_width=True):
            with st.spinner("Loading your data from database..."):
                df = load_transactions_from_db(st.session_state.user_id, limit=MAX_ROWS_IN_MEMORY)
                if df is not None and len(df) > 0:
                    st.session_state.csv_data = df
                    st.rerun()
                elif df is not None:
                    st.warning("⚠️ No data found in database for your account.")
                else:
                    st.error("❌ Failed to load data from database.")
    
    with col2:
        if st.button("📜 View Forecast History", use_container_width=True):
            forecast_history = get_forecast_history(st.session_state.user_id, limit=20)
            if forecast_history is not None and len(forecast_history) > 0:
                st.subheader("📜 Your Forecast History")
                st.dataframe(forecast_history, use_container_width=True, height=400)
            else:
                st.info("ℹ️ No forecast history found.")
    
    st.markdown("---")
    
    if st.session_state.csv_data is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"ℹ️ Current Dataset: **{len(st.session_state.csv_data):,}** baris")
        with col2:
            st.info(f"📋 Columns: **{len(st.session_state.csv_data.columns)}**")
        
        # Show data preview
        with st.expander("👁️ Quick Preview (10 rows)"):
            st.dataframe(st.session_state.csv_data.head(10), use_container_width=True)
    else:
        st.warning("⚠️ No data in current session. Upload CSV or load from database.")

def display_data_page():
    st.title("📊 Tampilkan Data")
    st.markdown("---")
    
    # Option to load from database
    if st.session_state.csv_data is None:
        st.info("ℹ️ No data loaded. You can load your data from database.")
        if st.button("📥 Load My Data from Database", type="primary"):
            with st.spinner("Loading your data..."):
                df = load_transactions_from_db(st.session_state.user_id, limit=MAX_ROWS_IN_MEMORY)
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
    
    # Option to load from database if no data
    if st.session_state.csv_data is None:
        st.info("ℹ️ No data loaded. Load your data first.")
        if st.button("📥 Load My Data from Database", type="primary"):
            with st.spinner("Loading your data..."):
                df = load_transactions_from_db(st.session_state.user_id, limit=MAX_ROWS_IN_MEMORY)
                if df is not None and len(df) > 0:
                    st.session_state.csv_data = df
                    st.rerun()
        return
    
    if st.session_state.csv_data is not None:
        total_rows = len(st.session_state.csv_data)
        st.info(f"ℹ️ Data saat ini: **{total_rows:,}** baris (Your personal data)")
        
        if st.button("🧹 Jalankan Advanced Cleaning", type="primary", use_container_width=True):
            with st.spinner("Membersihkan data dengan IQR method..."):
                try:
                    cleaned_df, cleaning_log = clean_data_advanced(st.session_state.csv_data)
                    st.session_state.cleaned_data = cleaned_df
                    st.success("✅ Data berhasil dibersihkan!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Rows", safe_metric_value(cleaning_log['original_rows']))
                    with col2:
                        st.metric("Final Rows", safe_metric_value(cleaning_log['final_rows']))
                    with col3:
                        st.metric("Removed", safe_metric_value(cleaning_log['total_removed']))
                    
                    st.subheader("📋 Cleaning Steps:")
                    for step in cleaning_log['steps']:
                        st.write(step)
                    
                    if 'outlier_info' in cleaning_log:
                        st.subheader("📊 Outlier Removal Details:")
                        for info in cleaning_log['outlier_info']:
                            st.write(f"**{info['column']}**: Removed {info['removed']} outliers "
                                   f"(Range: {info['lower_bound']:.2f} - {info['upper_bound']:.2f})")
                    
                    csv_cleaned = cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Cleaned Data",
                        data=csv_cleaned,
                        file_name=f"cleaned_no_outliers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Option to save cleaned data back to database
                    if st.button("💾 Save Cleaned Data to Database", type="secondary"):
                        with st.spinner("Saving cleaned data..."):
                            # First, delete old data for this user
                            conn = get_connection()
                            try:
                                with conn.cursor() as cursor:
                                    cursor.execute("DELETE FROM transactions WHERE user_id = %s", (st.session_state.user_id,))
                                conn.commit()
                            except Exception as e:
                                st.error(f"Error deleting old data: {e}")
                            finally:
                                conn.close()
                            
                            # Save cleaned data
                            inserted = save_transactions_to_db(cleaned_df, st.session_state.user_id)
                            if inserted > 0:
                                st.success(f"✅ Saved {inserted:,} cleaned rows to database!")
                    
                    gc.collect()
                except Exception as e:
                    st.error(f"❌ Error during cleaning: {str(e)}")
    else:
        st.warning("⚠️ Tidak ada data untuk dibersihkan. Upload file CSV terlebih dahulu.")

def add_data_page():
    st.title("➕ Add Data")
    st.markdown("---")
    tab1, tab2 = st.tabs(["📁 Upload CSV File", "🎲 Generate Sample Data"])
    
    with tab1:
        st.subheader("Upload CSV File (Support hingga 1GB)")
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
            if use_chunked:
                st.warning(f"⚡ File besar terdeteksi ({file_size_mb:.2f} MB). Menggunakan chunked loading otomatis.")
                chunk_size = st.number_input(
                    "Chunk size (baris per chunk):",
                    min_value=10000,
                    max_value=500000,
                    value=100000,
                    step=10000
                )
            
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
                        
                        if df is not None:
                            # ✅ FIX: Optimize memory before storing
                            df = optimize_dataframe_memory(df)
                            
                            # ✅ FIX: Warn if data is too large
                            if len(df) > MAX_ROWS_IN_MEMORY:
                                st.warning(f"⚠️ Large dataset ({len(df):,} rows). Consider saving to database and loading subsets.")
                            
                            st.session_state.csv_data = df
                            st.success(f"✅ Data loaded! Total: {len(df):,} baris | {len(df.columns)} kolom")
                            
                            with st.expander("👁️ Preview (10 baris)"):
                                st.dataframe(df.head(10), use_container_width=True)
                            
                            if auto_save_to_db:
                                with st.spinner("💾 Menyimpan ke database dengan user_id Anda..."):
                                    inserted = save_transactions_to_db(df, st.session_state.user_id)
                                    if inserted > 0:
                                        st.success(f"✅ Berhasil menyimpan {inserted:,} baris ke tabel `transactions` dengan user_id: {st.session_state.user_id}")
                                        
                                        # ✅ FIX: Clear memory if data saved to DB and too large
                                        if len(df) > MAX_ROWS_IN_MEMORY:
                                            st.info("💡 Data tersimpan di database. Clearing memory...")
                                            st.session_state.csv_data = None
                                            gc.collect()
                                    else:
                                        st.warning("⚠️ Tidak ada data yang disimpan ke database. Periksa format kolom.")
                            
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Tip: Untuk file sangat besar, coba tingkatkan chunk size atau pastikan memory cukup")
    
    with tab2:
        st.subheader("Generate Sample Data")
        st.warning(f"🔒 Sample data akan disimpan dengan user_id: {st.session_state.user_id}")
        
        sample_size = st.number_input(
            "Jumlah baris:",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
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
                    
                    if auto_save_sample:
                        with st.spinner("Saving to database..."):
                            inserted = save_transactions_to_db(sample_df, st.session_state.user_id)
                            if inserted > 0:
                                st.success(f"✅ Saved {inserted:,} rows to database with your user_id!")
                    
                    gc.collect()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

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
            with st.spinner("Loading your data..."):
                df = load_transactions_from_db(st.session_state.user_id, limit=MAX_ROWS_IN_MEMORY)
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
        st.error("Data tidak memiliki kolom tanggal yang terdeteksi. Pastikan ada kolom 'order_date' atau 'invoice_date'")
        return
    
    date_col = st.selectbox("Pilih Kolom Tanggal:", options=date_cols)
    value_options = ['Final_Price', 'price', 'subtotal'] + [col for col in data_source.columns if col not in date_cols]
    value_col = st.selectbox("Pilih Kolom Nilai:", options=value_options)
    
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"Generating {model_choice} forecast for {forecast_days} days..."):
            try:
                df_ts = data_source.copy()
                df_ts[date_col] = pd.to_datetime(df_ts[date_col])
                df_ts = df_ts.sort_values(date_col)
                df_ts.set_index(date_col, inplace=True)
                daily_ts = df_ts[value_col].resample('D').sum().fillna(0)
                
                st.info(f"📊 Time series created: {len(daily_ts)} days ({daily_ts.index.min()} to {daily_ts.index.max()})")
                
                test_size = min(30, len(daily_ts) // 5)
                train_size = len(daily_ts) - test_size
                train_data = daily_ts[:train_size]
                test_data = daily_ts[train_size:]
                
                st.write(f"📈 Training: {len(train_data)} days | Testing: {len(test_data)} days")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if model_choice == "SARIMA":
                    status_text.text("🔄 SARIMA Auto-Tuning: Mencari parameter terbaik...")
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
                    status_text.text(f"✅ Best model found: {summary['best_order']} x {summary['best_seasonal_order']}")
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
                else:
                    status_text.text("🔄 Training LSTM model...")
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
                status_text.text("✅ Forecast completed!")
                st.success(f"🎉 {model_choice} forecast berhasil di-generate!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.forecast_result:
        st.markdown("---")
        st.header("📊 Hasil Forecast")
        result = st.session_state.forecast_result
        st.subheader(f"🤖 Model: {result['model']}")
        
        if result['model'] == 'SARIMA':
            st.info(f"✅ Model: SARIMA {result['model_info']['best_order']} x {result['model_info']['best_seasonal_order']}")
        else:
            st.info(f"✅ Model: LSTM (Sequence Length: {result['model_info']['sequence_length']}, Units: {result['model_info']['lstm_units']})")
        
        st.markdown("---")
        
        # ✅ FIX: Sample data for plotting to avoid MessageSizeError
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        fig.suptitle(f'{result["model"]} Forecasting Results - Complete View', fontsize=16, fontweight='bold')
        
        recent_train = result['train_data'].tail(90)
        
        # ✅ FIX: Sample if too many points
        if len(recent_train) > MAX_PLOT_POINTS:
            sample_indices = np.linspace(0, len(recent_train)-1, MAX_PLOT_POINTS, dtype=int)
            recent_train_sampled = recent_train.iloc[sample_indices]
            ax.plot(recent_train_sampled.index, recent_train_sampled.values, 
                   label='Historical Data (Sampled)', linewidth=2.5, alpha=0.8)
            st.caption(f"📊 Chart displays {MAX_PLOT_POINTS} sampled points for optimal performance")
        else:
            ax.plot(recent_train.index, recent_train.values, 
                   label='Historical Data', linewidth=2.5, alpha=0.8)
        
        ax.plot(result['test_data'].index, result['test_data'].values, 
               label='Actual (Test Period)', linewidth=2.5, marker='o', markersize=4)
        ax.plot(result['test_data'].index, result['test_forecast'], 
               label='Forecast (Test Period)', linewidth=2, linestyle='--', marker='s', markersize=3)
        
        future_dates_with_connection = [result['train_data'].index[-1]] + list(result['future_dates'])
        future_values_with_connection = [result['train_data'].iloc[-1]] + list(result['future_forecast'])
        ax.plot(future_dates_with_connection, future_values_with_connection, 
               label=f'Future Forecast ({result["forecast_days"]} days)', 
               linewidth=2.5, linestyle='--', marker='o', markersize=4)
        
        ax.set_title('Complete Forecast Timeline', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', rotation=45)
        ax.axvline(x=result['train_data'].index[-1], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(result['train_data'].index[-1], ax.get_ylim()[1]*0.95, 'Today', 
               ha='center', fontsize=9, color='gray', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("📈 Forecast Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Forecast Period", f"{result['forecast_days']} days")
        with col2:
            avg_forecast = np.mean(result['future_forecast'])
            st.metric("Average Forecast", f"{avg_forecast:,.0f}")
        with col3:
            trend = "↗️ Increasing" if result['future_forecast'][-1] > result['future_forecast'][0] else "↘️ Decreasing"
            st.metric("Trend", trend)
        with col4:
            total_forecast = np.sum(result['future_forecast'])
            st.metric("Total Predicted", f"{total_forecast:,.0f}")
        
        st.markdown("---")
        st.subheader("📊 Model Performance Metrics")
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
        forecast_df = pd.DataFrame({
            'Date': result['future_dates'],
            'Forecast': result['future_forecast']
        })
        if result['model'] == 'SARIMA':
            forecast_df['Lower_CI'] = result['future_lower']
            forecast_df['Upper_CI'] = result['future_upper']
        
        csv_forecast = forecast_df.to_csv(index=False)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="⬇️ Download Forecast (CSV)",
                data=csv_forecast,
                file_name=f"forecast_{result['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        with col2:
            metrics_json = {
                'model': result['model'],
                'model_info': result['model_info'],
                'metrics': result['metrics'],
                'forecast_days': result['forecast_days'],
                'user_id': st.session_state.user_id
            }
            st.download_button(
                label="⬇️ Download Metrics (JSON)",
                data=str(metrics_json),
                file_name=f"metrics_{result['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        with col3:
            if st.button("💾 Save to Database"):
                with st.spinner("Saving forecast to database..."):
                    inserted = save_forecasts_to_db(
                        st.session_state.user_id,
                        result['model'], 
                        result['future_dates'], 
                        result['future_forecast']
                    )
                    if inserted > 0:
                        st.success(f"✅ Saved {inserted:,} forecast records with user_id: {st.session_state.user_id}")
                    else:
                        st.error("❌ Failed to save forecast to database.")
        
        with st.expander("📋 View Forecast Table"):
            # ✅ FIX: Limit table display
            st.dataframe(forecast_df, use_container_width=True, height=400)

# ------------------ MAIN ------------------
def main():
    # Check schema on first load
    if 'schema_checked' not in st.session_state:
        check_and_update_schema()
        st.session_state.schema_checked = True
    
    if not st.session_state.logged_in:
        login_page()
    else:
        with st.sidebar:
            st.title("PT. XYZ")
            st.markdown("**Dashboard System**")
            st.markdown("---")
            st.write(f"👤 **{st.session_state.username}**")
            st.caption(f"User ID: {st.session_state.user_id}")
            
            # Display user statistics
            user_stats = get_user_statistics(st.session_state.user_id)
            st.markdown("---")
            st.subheader("📊 Your Statistics")
            
            # ✅ FIX: Use safe_metric_value
            st.metric("DB Transactions", safe_metric_value(user_stats['total_transactions']))
            st.metric("DB Forecasts", safe_metric_value(user_stats['total_forecasts']))
            
            if st.session_state.csv_data is not None:
                st.markdown("---")
                st.subheader("💾 Session Data")
                st.metric("Memory Rows", safe_metric_value(len(st.session_state.csv_data)))
                memory_mb = st.session_state.csv_data.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory Usage", f"{memory_mb:.2f} MB")
                if st.session_state.cleaned_data is not None:
                    st.metric("Cleaned Rows", safe_metric_value(len(st.session_state.cleaned_data)))
            
            st.markdown("---")
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
                label_visibility="visible"
            )
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True, type="primary"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.user_id = None
                st.session_state.csv_data = None
                st.session_state.cleaned_data = None
                st.session_state.forecast_result = None
                gc.collect()
                st.rerun()
        
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

if __name__ == "__main__":
    main()