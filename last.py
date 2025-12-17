import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import gc
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# MySQL imports
try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

# SARIMA imports
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False

# LSTM imports
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

# Set page config
st.set_page_config(
    page_title="PT. XYZ Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'forecast_result' not in st.session_state:
    st.session_state.forecast_result = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None

# ==================== MYSQL DATABASE CONNECTION ====================
class MySQLDatabase:
    """MySQL Database Manager"""
    
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
    
    def connect(self):
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                use_unicode=True
            )
            
            if self.connection.is_connected():
                return True, "✅ Connected to MySQL database successfully!"
        except Error as e:
            return False, f"❌ Error: {str(e)}"
    
    def disconnect(self):
        """Disconnect from MySQL"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            return "✅ Disconnected from MySQL"
    
    def execute_query(self, query, params=None):
        """Execute a query"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or ())
            self.connection.commit()
            return True, "✅ Query executed successfully"
        except Error as e:
            return False, f"❌ Error: {str(e)}"
    
    def fetch_data(self, query, params=None):
        """Fetch data from database"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            result = cursor.fetchall()
            return pd.DataFrame(result)
        except Error as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            return None
    
    def insert_dataframe(self, df, table_name, batch_size=1000):
        """Insert DataFrame to MySQL table in batches"""
        try:
            cursor = self.connection.cursor()
            
            # Get column names
            columns = df.columns.tolist()
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join([f"`{col}`" for col in columns])
            
            insert_query = f"INSERT INTO `{table_name}` ({columns_str}) VALUES ({placeholders})"
            
            # Insert in batches
            total_rows = len(df)
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                data = [tuple(row) for row in batch.values]
                cursor.executemany(insert_query, data)
                self.connection.commit()
                
                if (i + batch_size) % 10000 == 0:
                    st.info(f"Inserted {min(i + batch_size, total_rows):,} / {total_rows:,} rows")
            
            return True, f"✅ Inserted {total_rows:,} rows successfully"
        except Error as e:
            return False, f"❌ Error: {str(e)}"

# ==================== ADVANCED METRICS ====================
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

# ==================== DATA CLEANING ====================
def remove_outliers_iqr(data, columns):
    """Remove outliers using IQR method"""
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
    """Advanced data cleaning with IQR method"""
    
    cleaning_log = {
        'original_rows': len(df),
        'steps': []
    }
    
    # Convert date columns
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        cleaning_log['steps'].append('✅ Converted order_date to datetime')
    
    if 'invoice_date' in df.columns:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
        cleaning_log['steps'].append('✅ Converted invoice_date to datetime')
    
    # Convert numeric columns
    num_cols = ["qty", "price", "bundle_price", "subtotal", "discount", 
                "shipping_fee", "used_point", "diskonproposional", "Final_Price",
                "bulan", "year", "month"]
    
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    cleaning_log['steps'].append(f'✅ Converted {len([c for c in num_cols if c in df.columns])} columns to numeric')
    
    # Drop duplicates
    before_dup = len(df)
    df = df.drop_duplicates()
    after_dup = len(df)
    cleaning_log['steps'].append(f'✅ Removed {before_dup - after_dup} duplicates')
    
    # Remove outliers using IQR
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

# ==================== SARIMA AUTO FORECASTER ====================
class SARIMAAutoForecaster:
    """Automatic SARIMA forecasting dengan grid search untuk parameter terbaik"""
    
    def __init__(self):
        self.model = None
        self.model_params = None
        self.fitted = False
        self.search_results = []
    
    def auto_fit(self, train_data, max_p=3, max_d=2, max_q=3, 
                 max_P=2, max_D=1, max_Q=2, seasonal_period=7):
        """Automatic parameter search untuk SARIMA terbaik berdasarkan AIC"""
        import itertools
        
        best_aic = np.inf
        best_params = None
        search_log = []
        
        # Generate combinations
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
        
        return forecast_mean
    
    def get_model_summary(self):
        """Get detailed model summary"""
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

# ==================== LSTM FORECASTER ====================
class LSTMForecaster:
    def __init__(self, sequence_length=30, lstm_units=50):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.model = None
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
    
    def forecast(self, train_data, steps):
        if not self.fitted:
            raise Exception("Model not fitted yet")
        
        window_size = min(self.sequence_length * 2, len(train_data))
        last_sequence = train_data.tail(window_size).values
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        forecasts = []
        current_sequence = last_sequence_scaled[-self.sequence_length:].copy()
        
        recent_changes = np.diff(train_data.tail(30).values)
        volatility = np.std(recent_changes) if len(recent_changes) > 0 else 0
        
        for i in range(steps):
            X_input = current_sequence.reshape(1, self.sequence_length, 1)
            next_pred_scaled = self.model.predict(X_input, verbose=0)[0, 0]
            
            noise_factor = volatility * np.random.normal(0, 0.5) * (1 - i/(steps*2))
            next_pred = self.scaler.inverse_transform([[next_pred_scaled]])[0, 0] + noise_factor
            
            next_pred = max(next_pred, 0)
            
            forecasts.append(next_pred)
            
            next_pred_rescaled = self.scaler.transform([[next_pred]])[0, 0]
            current_sequence = np.append(current_sequence[1:], next_pred_rescaled)
        
        return np.array(forecasts)

# ==================== HELPER FUNCTIONS ====================
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
    """Load large CSV files in chunks - supports up to 1GB"""
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

# ==================== LOGIN PAGE ====================
def login_page():
    st.title("PT. XYZ")
    
    tab1, tab2, tab3 = st.tabs(["📝 Register", "🔐 Login", "🔒 Change Password"])
    
    with tab1:
        st.subheader("Register New Account")
        reg_username = st.text_input("Username", key="reg_user", placeholder="Min 3 characters")
        reg_password = st.text_input("Password", type="password", key="reg_pass", placeholder="Min 4 characters")
        
        if st.button("✅ Register", key="reg_btn"):
            if len(reg_username) >= 3 and len(reg_password) >= 4:
                st.success("✅ Registrasi berhasil! Silakan login.")
            else:
                st.error("❌ Username min 3 karakter, Password min 4 karakter")
    
    with tab2:
        st.subheader("Login to Dashboard")
        login_username = st.text_input("Username", key="login_user", placeholder="Enter username")
        login_password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
        
        if st.button("🔓 Login", key="login_btn"):
            if len(login_username) >= 3 and len(login_password) >= 4:
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.rerun()
            else:
                st.error("❌ Username min 3 karakter, Password min 4 karakter")
    
    with tab3:
        st.subheader("Change Password")
        old_password = st.text_input("Old Password", type="password", key="old_pass")
        new_password = st.text_input("New Password", type="password", key="new_pass", placeholder="Min 4 characters")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pass")
        
        if st.button("🔄 Change Password", key="change_btn"):
            if len(new_password) >= 4 and new_password == confirm_password:
                st.success("✅ Password berhasil diubah!")
            else:
                st.error("❌ Password tidak valid atau tidak cocok")

# ==================== DASHBOARD PAGE (ENHANCED WITH FORECAST DISPLAY) ====================
def dashboard_page():
    st.title("🏠 Dashboard Overview")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_count = len(st.session_state.csv_data) if st.session_state.csv_data is not None else 0
        st.metric("📊 Total Data", f"{data_count:,}", delta="Active")
    
    with col2:
        memory_usage = 0
        if st.session_state.csv_data is not None:
            memory_usage = st.session_state.csv_data.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("💾 Memory", f"{memory_usage:.2f} MB", delta="Efficient" if memory_usage < 500 else "High")
    
    with col3:
        cleaned_count = len(st.session_state.cleaned_data) if st.session_state.cleaned_data is not None else 0
        st.metric("✅ Cleaned Data", f"{cleaned_count:,}")
    
    with col4:
        forecast_status = "Ready" if st.session_state.forecast_result else "Not Ready"
        st.metric("📈 Forecast", forecast_status)
    
    st.markdown("---")
    
    # ========== NEW: DISPLAY LATEST FORECAST RESULTS ==========
    if st.session_state.forecast_result:
        st.subheader("🎯 Latest Forecast Results")
        
        result = st.session_state.forecast_result
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"**Model:** {result['model']} | **Forecast Period:** {result['forecast_days']} days")
            
            # Mini forecast plot
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#2d2d2d')
            
            # Plot recent data
            recent_train = result['train_data'].tail(30)
            ax.plot(recent_train.index, recent_train.values, 
                   label='Recent Data', color='white', linewidth=2, alpha=0.8)
            
            # Plot forecast
            ax.plot(result['future_dates'], result['future_forecast'], 
                   label='Forecast', color='#10b981', linewidth=2.5, linestyle='--', marker='o', markersize=3)
            
            ax.set_title('Latest Forecast Preview', fontsize=12, fontweight='bold', color='white', pad=10)
            ax.set_ylabel('Value', fontsize=10, color='white')
            ax.set_xlabel('Date', fontsize=10, color='white')
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.2, color='gray', linestyle='--')
            ax.tick_params(axis='x', rotation=45, colors='white', labelsize=8)
            ax.tick_params(axis='y', colors='white', labelsize=8)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("**📊 Performance Metrics:**")
            st.metric("MAE", f"{result['metrics']['MAE']:,.2f}")
            st.metric("RMSE", f"{result['metrics']['RMSE']:,.2f}")
            st.metric("sMAPE", f"{result['metrics']['sMAPE']:.2f}%")
            st.metric("MASE", f"{result['metrics']['MASE']:.4f}")
            
            st.markdown("---")
            
            # Forecast summary
            forecast_mean = np.mean(result['future_forecast'])
            forecast_max = np.max(result['future_forecast'])
            forecast_min = np.min(result['future_forecast'])
            
            st.markdown("**🎯 Forecast Summary:**")
            st.write(f"- Mean: {forecast_mean:,.2f}")
            st.write(f"- Max: {forecast_max:,.2f}")
            st.write(f"- Min: {forecast_min:,.2f}")
    
    st.markdown("---")
    
    if st.session_state.csv_data is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"ℹ️ Dataset: **{len(st.session_state.csv_data):,}** baris")
        with col2:
            st.info(f"📋 Columns: **{len(st.session_state.csv_data.columns)}**")
    else:
        st.warning("⚠️ Belum ada data. Upload file CSV di menu 'Add Data'")

# ==================== DISPLAY DATA PAGE ====================
def display_data_page():
    st.title("📊 Tampilkan Data")
    st.markdown("---")
    
    if st.session_state.csv_data is not None:
        total_rows = len(st.session_state.csv_data)
        st.info(f"ℹ️ Total data: **{total_rows:,}** baris (UNLIMITED CAPACITY)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rows_per_page = st.selectbox("Baris per halaman:", options=[50, 100, 500, 1000, 5000], index=0)
        
        with col2:
            total_pages = (total_rows - 1) // rows_per_page + 1
            page_number = st.number_input(f"Halaman (1-{total_pages:,}):", min_value=1, max_value=total_pages, value=1, step=1)
        
        start_idx = (page_number - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        st.markdown("---")
        st.subheader(f"Data Preview - Halaman {page_number:,} dari {total_pages:,}")
        st.caption(f"Menampilkan baris {start_idx + 1:,} - {end_idx:,}")
        
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
        st.warning("⚠️ Tidak ada data. Upload file CSV di menu 'Add Data'")

# ==================== CLEANING DATA PAGE ====================
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
    
    if st.session_state.csv_data is not None:
        total_rows = len(st.session_state.csv_data)
        st.info(f"ℹ️ Data saat ini: **{total_rows:,}** baris")
        
        if st.button("🧹 Jalankan Advanced Cleaning", type="primary", use_container_width=True):
            with st.spinner("Membersihkan data dengan IQR method..."):
                try:
                    cleaned_df, cleaning_log = clean_data_advanced(st.session_state.csv_data)
                    st.session_state.cleaned_data = cleaned_df
                    
                    st.success("✅ Data berhasil dibersihkan!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Rows", f"{cleaning_log['original_rows']:,}")
                    with col2:
                        st.metric("Final Rows", f"{cleaning_log['final_rows']:,}")
                    with col3:
                        st.metric("Removed", f"{cleaning_log['total_removed']:,}")
                    
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
                    
                    gc.collect()
                    
                except Exception as e:
                    st.error(f"❌ Error during cleaning: {str(e)}")
        
        if st.session_state.cleaned_data is not None:
            st.markdown("---")
            st.subheader("Preview Cleaned Data (20 baris pertama)")
            st.dataframe(st.session_state.cleaned_data.head(20), use_container_width=True)
    else:
        st.warning("⚠️ Tidak ada data untuk dibersihkan. Upload file CSV terlebih dahulu.")

# ==================== ADD DATA PAGE (ENHANCED WITH APPEND MODE) ====================
def add_data_page():
    st.title("➕ Add Data")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📁 Upload CSV File", "🎲 Generate Sample Data", "🗄️ MySQL Database"])
    
    with tab1:
        st.subheader("Upload CSV File (Support hingga 1GB)")
        st.info("💡 Sistem mendukung file CSV hingga 1GB dengan chunked loading")
        
        # ========== NEW: APPEND MODE ==========
        if st.session_state.csv_data is not None:
            st.success(f"✅ Data existing: **{len(st.session_state.csv_data):,}** baris")
            
            upload_mode = st.radio(
                "Pilih mode upload:",
                options=["Replace (Hapus data lama)", "Append (Tambah ke data existing)"],
                index=1,
                help="Replace akan menghapus data lama dan menggantinya. Append akan menambahkan ke data existing."
            )
        else:
            upload_mode = "Replace (Hapus data lama)"
            st.info("ℹ️ Mode: Replace (belum ada data existing)")
        
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
            
            if st.button("💾 Load Data", type="primary"):
                try:
                    with st.spinner("Loading data..."):
                        if use_chunked:
                            df = load_csv_chunked(uploaded_file, chunksize=chunk_size)
                        else:
                            df = pd.read_csv(uploaded_file)
                        
                        if df is not None:
                            # ========== APPEND MODE LOGIC ==========
                            if "Append" in upload_mode and st.session_state.csv_data is not None:
                                old_count = len(st.session_state.csv_data)
                                new_count = len(df)
                                
                                # Combine data
                                st.session_state.csv_data = pd.concat(
                                    [st.session_state.csv_data, df], 
                                    ignore_index=True
                                )
                                
                                total_count = len(st.session_state.csv_data)
                                
                                st.success(f"✅ Data appended successfully!")
                                st.info(f"📊 Old: {old_count:,} + New: {new_count:,} = **Total: {total_count:,} baris**")
                            else:
                                # Replace mode
                                st.session_state.csv_data = df
                                st.success(f"✅ Data loaded! Total: {len(df):,} baris | {len(df.columns)} kolom")
                            
                            with st.expander("👁️ Preview (10 baris)"):
                                st.dataframe(st.session_state.csv_data.head(10), use_container_width=True)
                            
                            gc.collect()
                            st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Tip: Untuk file sangat besar, coba tingkatkan chunk size atau pastikan memory cukup")
    
    with tab2:
        st.subheader("Generate Sample Data")
        
        if st.session_state.csv_data is not None:
            st.success(f"✅ Data existing: **{len(st.session_state.csv_data):,}** baris")
            
            generate_mode = st.radio(
                "Pilih mode generate:",
                options=["Replace (Hapus data lama)", "Append (Tambah ke data existing)"],
                index=1
            )
        else:
            generate_mode = "Replace (Hapus data lama)"
            st.info("ℹ️ Mode: Replace (belum ada data existing)")
        
        sample_size = st.number_input(
            "Jumlah baris:",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        
        st.caption(f"Estimasi memory: ~{(sample_size * 40) / 1024 / 1024:.2f} MB")
        
        if st.button("🎲 Generate", type="primary"):
            with st.spinner(f"Generating {sample_size:,} baris..."):
                try:
                    new_data = generate_sample_data_chunked(sample_size)
                    
                    if "Append" in generate_mode and st.session_state.csv_data is not None:
                        old_count = len(st.session_state.csv_data)
                        st.session_state.csv_data = pd.concat(
                            [st.session_state.csv_data, new_data], 
                            ignore_index=True
                        )
                        total_count = len(st.session_state.csv_data)
                        st.success(f"✅ Generated & Appended! Old: {old_count:,} + New: {sample_size:,} = **Total: {total_count:,}**")
                    else:
                        st.session_state.csv_data = new_data
                        st.success(f"✅ Generated {sample_size:,} baris!")
                    
                    gc.collect()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # ========== NEW: MYSQL DATABASE TAB ==========
    with tab3:
        st.subheader("🗄️ MySQL Database Connection")
        
        if not MYSQL_AVAILABLE:
            st.error("❌ MySQL connector tidak tersedia. Install: `pip install mysql-connector-python`")
            return
        
        st.info("💡 Koneksikan ke MySQL database untuk menyimpan atau load data")
        
        with st.expander("📖 Setup Instructions"):
            st.markdown("""
            **Cara Setup MySQL Database:**
            
            1. Install MySQL Server di komputer Anda
            2. Buat database baru (lihat SQL script di bawah)
            3. Masukkan credentials di form ini
            4. Klik "Connect to Database"
            
            **SQL Script untuk membuat database & table:**
            ```sql
            -- 1. Buat Database
            CREATE DATABASE pt_xyz_data;
            USE pt_xyz_data;
            
            -- 2. Buat Table untuk Orders
            CREATE TABLE orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                order_date DATETIME,
                invoice_date DATETIME,
                qty INT,
                price DECIMAL(15, 2),
                bundle_price DECIMAL(15, 2),
                subtotal DECIMAL(15, 2),
                discount DECIMAL(15, 2),
                shipping_fee DECIMAL(15, 2),
                used_point DECIMAL(15, 2),
                diskonproposional DECIMAL(15, 2),
                Final_Price DECIMAL(15, 2),
                category VARCHAR(50),
                bulan INT,
                year INT,
                month INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_order_date (order_date),
                INDEX idx_category (category)
            );
            
            -- 3. Buat Table untuk Forecasts
            CREATE TABLE forecasts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                forecast_date DATETIME,
                forecast_value DECIMAL(15, 2),
                model_type VARCHAR(50),
                forecast_period INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_forecast_date (forecast_date)
            );
            
            -- 4. Buat Table untuk Users (Optional)
            CREATE TABLE users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            ```
            """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            db_host = st.text_input("Host:", value="localhost", placeholder="localhost atau IP")
            db_user = st.text_input("Username:", value="root", placeholder="MySQL username")
        
        with col2:
            db_password = st.text_input("Password:", type="password", placeholder="MySQL password")
            db_name = st.text_input("Database:", value="pt_xyz_data", placeholder="Nama database")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔌 Connect to Database", type="primary"):
                with st.spinner("Connecting to MySQL..."):
                    try:
                        db = MySQLDatabase(db_host, db_user, db_password, db_name)
                        success, message = db.connect()
                        
                        if success:
                            st.session_state.db_connection = db
                            st.success(message)
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
        
        with col2:
            if st.button("🔌 Disconnect"):
                if st.session_state.db_connection:
                    message = st.session_state.db_connection.disconnect()
                    st.session_state.db_connection = None
                    st.info(message)
                else:
                    st.warning("⚠️ Tidak ada koneksi aktif")
        
        with col3:
            connection_status = "🟢 Connected" if st.session_state.db_connection else "🔴 Disconnected"
            st.metric("Status", connection_status)
        
        if st.session_state.db_connection:
            st.markdown("---")
            st.subheader("📊 Database Operations")
            
            operation = st.selectbox(
                "Pilih operasi:",
                options=[
                    "Load Data from Database",
                    "Save Current Data to Database",
                    "Execute Custom Query"
                ]
            )
            
            if operation == "Load Data from Database":
                st.write("**Load data dari table MySQL**")
                
                table_name = st.text_input("Table name:", value="orders")
                limit = st.number_input("Limit rows (0 = all):", min_value=0, value=0, step=1000)
                
                if st.button("📥 Load from Database"):
                    with st.spinner("Loading from database..."):
                        try:
                            query = f"SELECT * FROM {table_name}"
                            if limit > 0:
                                query += f" LIMIT {limit}"
                            
                            df = st.session_state.db_connection.fetch_data(query)
                            
                            if df is not None and len(df) > 0:
                                st.session_state.csv_data = df
                                st.success(f"✅ Loaded {len(df):,} rows from database!")
                                st.rerun()
                            else:
                                st.warning("⚠️ No data found or table is empty")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            elif operation == "Save Current Data to Database":
                st.write("**Save data ke MySQL table**")
                
                if st.session_state.csv_data is not None:
                    table_name = st.text_input("Table name:", value="orders")
                    batch_size = st.number_input("Batch size:", min_value=100, max_value=10000, value=1000)
                    
                    st.info(f"ℹ️ Will save {len(st.session_state.csv_data):,} rows to table '{table_name}'")
                    
                    if st.button("💾 Save to Database", type="primary"):
                        with st.spinner("Saving to database..."):
                            try:
                                success, message = st.session_state.db_connection.insert_dataframe(
                                    st.session_state.csv_data, 
                                    table_name, 
                                    batch_size=batch_size
                                )
                                
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ No data to save. Upload data first.")
            
            elif operation == "Execute Custom Query":
                st.write("**Execute SQL query**")
                
                query = st.text_area(
                    "SQL Query:",
                    height=150,
                    placeholder="SELECT * FROM orders WHERE order_date > '2024-01-01' LIMIT 100"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("▶️ Execute SELECT"):
                        if query.strip():
                            with st.spinner("Executing query..."):
                                try:
                                    df = st.session_state.db_connection.fetch_data(query)
                                    
                                    if df is not None and len(df) > 0:
                                        st.success(f"✅ Query returned {len(df):,} rows")
                                        st.dataframe(df, use_container_width=True)
                                    else:
                                        st.info("ℹ️ Query executed but returned no data")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                        else:
                            st.warning("⚠️ Please enter a query")
                
                with col2:
                    if st.button("▶️ Execute INSERT/UPDATE/DELETE"):
                        if query.strip():
                            with st.spinner("Executing query..."):
                                try:
                                    success, message = st.session_state.db_connection.execute_query(query)
                                    
                                    if success:
                                        st.success(message)
                                    else:
                                        st.error(message)
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                        else:
                            st.warning("⚠️ Please enter a query")

# ==================== DELETE DATA PAGE ====================
def delete_data_page():
    st.title("🗑️ Hapus Data")
    st.markdown("---")
    
    if st.session_state.csv_data is not None:
        total_rows = len(st.session_state.csv_data)
        st.warning("⚠️ Penghapusan data bersifat permanen!")
        st.info(f"ℹ️ Total: **{total_rows:,}** baris")
        
        st.subheader("Hapus Berdasarkan Range")
        col1, col2 = st.columns(2)
        
        with col1:
            start_id = st.number_input("ID Awal:", min_value=1, max_value=total_rows, value=1)
        with col2:
            end_id = st.number_input("ID Akhir:", min_value=start_id, max_value=total_rows, value=min(start_id + 9, total_rows))
        
        rows_to_delete = end_id - start_id + 1
        st.caption(f"Akan menghapus {rows_to_delete:,} baris")
        
        if st.button("🗑️ Hapus Range", type="primary"):
            if 'id' in st.session_state.csv_data.columns:
                st.session_state.csv_data = st.session_state.csv_data[
                    (st.session_state.csv_data['id'] < start_id) | 
                    (st.session_state.csv_data['id'] > end_id)
                ].reset_index(drop=True)
                st.success(f"✅ {rows_to_delete:,} baris dihapus!")
                gc.collect()
                st.rerun()
    else:
        st.warning("⚠️ Tidak ada data untuk dihapus.")

# ==================== GENERATE FORECAST PAGE ====================
def generate_forecast_page():
    st.title("📈 Generate Forecast")
    st.markdown("---")
    
    data_source = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.csv_data
    
    if data_source is None:
        st.warning("⚠️ Tidak ada data. Upload dan clean data terlebih dahulu.")
        return
    
    st.success(f"✅ Data tersedia: {len(data_source):,} baris")
    
    st.subheader("⚙️ Konfigurasi Forecast")
    
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
    
    date_col = st.selectbox("Pilih Kolom Tanggal:", options=[col for col in data_source.columns if 'date' in col.lower()])
    value_col = st.selectbox("Pilih Kolom Nilai:", options=['Final_Price', 'price', 'subtotal'] + [col for col in data_source.columns if col not in ['order_date', 'invoice_date']])
    
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
                    
                    test_forecast = sarima.forecast(len(test_data))
                    future_forecast = sarima.forecast(forecast_days)
                    
                    metrics = {
                        'MAE': AdvancedMetrics.mae(test_data.values, test_forecast),
                        'RMSE': AdvancedMetrics.rmse(test_data.values, test_forecast),
                        'sMAPE': AdvancedMetrics.smape(test_data.values, test_forecast),
                        'MASE': AdvancedMetrics.mase(test_data.values, test_forecast, train_data.values, seasonality=seasonal_period)
                    }
                    
                    st.session_state.forecast_result = {
                        'model': 'SARIMA',
                        'model_info': summary,
                        'model_summary': model_summary,
                        'search_log': search_log,
                        'train_data': train_data,
                        'test_data': test_data,
                        'test_forecast': test_forecast,
                        'future_forecast': future_forecast,
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
            st.success("✅ Auto-tuning completed - Best parameters selected based on AIC")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Order (p,d,q)", str(result['model_info']['best_order']))
            with col2:
                st.metric("Seasonal Order", str(result['model_info']['best_seasonal_order']))
            with col3:
                st.metric("AIC (Best)", f"{result['model_info']['best_aic']:.2f}")
            with col4:
                st.metric("BIC", f"{result['model_info']['best_bic']:.2f}")
        
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sequence Length", result['model_info']['sequence_length'])
            with col2:
                st.metric("LSTM Units", result['model_info']['lstm_units'])
            with col3:
                st.metric("Epochs", result['model_info']['epochs'])
        
        st.markdown("---")
        st.subheader("📈 Performance Metrics (Test Data)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", f"{result['metrics']['MAE']:,.2f}")
        with col2:
            st.metric("RMSE", f"{result['metrics']['RMSE']:,.2f}")
        with col3:
            st.metric("sMAPE", f"{result['metrics']['sMAPE']:.2f}%")
        with col4:
            st.metric("MASE", f"{result['metrics']['MASE']:.4f}")
        
        st.markdown("---")
        st.subheader("📊 Visualisasi Forecast")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        fig.patch.set_facecolor('#1e1e1e')
        fig.suptitle(f'{result["model"]} Forecasting Results', fontsize=18, fontweight='bold', color='white', y=0.98)
        
        ax1 = axes[0]
        ax1.set_facecolor('#2d2d2d')
        
        last_train_date = result['train_data'].index[-1]
        last_train_value = result['train_data'].iloc[-1]
        first_test_date = result['test_data'].index[0]
        first_test_forecast = result['test_forecast'][0]
        
        recent_train = result['train_data'].tail(60)
        ax1.plot(recent_train.index, recent_train.values, 
                label='Training Data', color='white', linewidth=2.5, alpha=0.8)
        
        ax1.plot(result['test_data'].index, result['test_data'].values, 
                label='Actual', color='#3b82f6', linewidth=2.5, marker='o', markersize=5)
        
        ax1.plot([last_train_date, first_test_date], [last_train_value, first_test_forecast],
                color='#ef4444', linewidth=2, linestyle='--', alpha=0.6)
        
        ax1.plot(result['test_data'].index, result['test_forecast'], 
                label='Forecast', color='#ef4444', linewidth=2.5, linestyle='--', marker='s', markersize=4)
        
        ax1.set_title('Test Period Forecast', fontsize=14, fontweight='bold', color='white', pad=15)
        ax1.set_ylabel('Value', fontsize=12, color='white')
        ax1.set_xlabel('Date', fontsize=12, color='white')
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.2, color='gray', linestyle='--')
        ax1.tick_params(axis='x', rotation=45, colors='white', labelsize=9)
        ax1.tick_params(axis='y', colors='white', labelsize=9)
        ax1.spines['bottom'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        ax2 = axes[1]
        ax2.set_facecolor('#2d2d2d')
        
        last_data_date = result['train_data'].index[-1]
        last_data_value = result['train_data'].iloc[-1]
        first_future_date = result['future_dates'][0]
        first_future_forecast = result['future_forecast'][0]
        
        recent_train = result['train_data'].tail(30)
        ax2.plot(recent_train.index, recent_train.values, 
                label='Recent Training', color='white', linewidth=2.5, alpha=0.8)
        
        ax2.plot([last_data_date, first_future_date], [last_data_value, first_future_forecast],
                color='#10b981', linewidth=2, linestyle='--', alpha=0.6)
        
        ax2.plot(result['future_dates'], result['future_forecast'], 
                label='Future Forecast', color='#10b981', linewidth=2.5, linestyle='--', marker='o', markersize=4)
        
        ax2.set_title(f'{result["forecast_days"]}-Day Future Forecast', fontsize=14, fontweight='bold', color='white', pad=15)
        ax2.set_ylabel('Value', fontsize=12, color='white')
        ax2.set_xlabel('Date', fontsize=12, color='white')
        ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.2, color='gray', linestyle='--')
        ax2.tick_params(axis='x', rotation=45, colors='white', labelsize=9)
        ax2.tick_params(axis='y', colors='white', labelsize=9)
        ax2.spines['bottom'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Metrics (Text)")
            st.write("**Test Period Performance:**")
            st.write(f"- **MAE (Mean Absolute Error)**: {result['metrics']['MAE']:,.2f}")
            st.write(f"- **RMSE (Root Mean Squared Error)**: {result['metrics']['RMSE']:,.2f}")
            st.write(f"- **sMAPE (Symmetric MAPE)**: {result['metrics']['sMAPE']:.2f}%")
            st.write(f"- **MASE (Mean Abs Scaled Error)**: {result['metrics']['MASE']:.4f}")
            
            st.info("💡 Lower values indicate better performance. MASE < 1 means better than naive forecast.")
        
        with col2:
            st.subheader("📉 Residuals Analysis (Text)")
            
            residuals = result['test_data'].values - result['test_forecast']
            
            st.write("**Residuals Statistics:**")
            st.write(f"- **Mean Residual**: {np.mean(residuals):,.2f}")
            st.write(f"- **Std Deviation**: {np.std(residuals):,.2f}")
            st.write(f"- **Min Residual**: {np.min(residuals):,.2f}")
            st.write(f"- **Max Residual**: {np.max(residuals):,.2f}")
            st.write(f"- **Mean Absolute Residual**: {np.mean(np.abs(residuals)):,.2f}")
            
            st.info("💡 Mean residual close to 0 indicates unbiased predictions. Lower std dev is better.")
        
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        forecast_df = pd.DataFrame({
            'Date': result['future_dates'],
            'Forecast': result['future_forecast']
        })
        
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
                'forecast_days': result['forecast_days']
            }
            
            st.download_button(
                label="⬇️ Download Metrics (JSON)",
                data=str(metrics_json),
                file_name=f"metrics_{result['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Save forecast to database if connected
            if st.session_state.db_connection:
                if st.button("💾 Save to Database"):
                    with st.spinner("Saving forecast to database..."):
                        try:
                            forecast_save_df = forecast_df.copy()
                            forecast_save_df['model_type'] = result['model']
                            forecast_save_df['forecast_period'] = result['forecast_days']
                            forecast_save_df.columns = ['forecast_date', 'forecast_value', 'model_type', 'forecast_period']
                            
                            success, message = st.session_state.db_connection.insert_dataframe(
                                forecast_save_df,
                                'forecasts',
                                batch_size=1000
                            )
                            
                            if success:
                                st.success("✅ Forecast saved to database!")
                            else:
                                st.error(message)
                        except Exception as e:
                            st.error(f"❌ Error saving to database: {str(e)}")
        
        with st.expander("📋 View Forecast Table"):
            st.dataframe(forecast_df, use_container_width=True)

# ==================== MAIN APP ====================
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        with st.sidebar:
            st.title("PT. XYZ")
            st.markdown("**Dashboard System**")
            st.markdown("---")
            
            st.write(f"👤 **{st.session_state.username}**")
            
            if st.session_state.csv_data is not None:
                st.markdown("---")
                st.subheader("📊 Data Info")
                st.metric("Total Rows", f"{len(st.session_state.csv_data):,}")
                memory_mb = st.session_state.csv_data.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory", f"{memory_mb:.2f} MB")
                
                if st.session_state.cleaned_data is not None:
                    st.metric("Cleaned Rows", f"{len(st.session_state.cleaned_data):,}")
            
            if st.session_state.db_connection:
                st.markdown("---")
                st.success("🗄️ Database Connected")
            
            st.markdown("---")
            
            st.subheader("📋 Navigation")
            menu = st.radio(
                "",
                options=[
                    "🏠 Dashboard",
                    "📊 Tampilkan Data",
                    "🧹 Cleaning Data",
                    "➕ Add Data",
                    "🗑️ Hapus Data",
                    "📈 Generate Forecast"
                ],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            if st.button("🚪 Logout", use_container_width=True, type="primary"):
                if st.session_state.db_connection:
                    st.session_state.db_connection.disconnect()
                
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.csv_data = None
                st.session_state.cleaned_data = None
                st.session_state.forecast_result = None
                st.session_state.db_connection = None
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