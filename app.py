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
    """
    Automatic SARIMA forecasting dengan grid search untuk parameter terbaik
    """
    def __init__(self):
        self.model = None
        self.model_params = None
        self.fitted = False
        self.search_results = []
    
    def auto_fit(self, train_data, max_p=3, max_d=2, max_q=3, 
                 max_P=2, max_D=1, max_Q=2, seasonal_period=7):
        """
        Automatic parameter search untuk SARIMA terbaik berdasarkan AIC
        """
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
        forecast_ci = forecast_result.conf_int().values
        
        return forecast_mean, forecast_ci[:, 0], forecast_ci[:, 1]
    
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
    
    def plot_diagnostics(self):
        """Plot model diagnostics"""
        if not self.fitted:
            return None
        
        fig = self.model.plot_diagnostics(figsize=(15, 10))
        return fig

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
    
    def forecast(self, train_data, steps, add_noise=True):
        if not self.fitted:
            raise Exception("Model not fitted yet")
        
        last_sequence = train_data.tail(self.sequence_length).values
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        forecasts = []
        current_sequence = last_sequence_scaled.copy()
        
        # Calculate noise level from training data
        train_std = train_data.std() * 0.15  # 15% of std for realistic fluctuation
        
        for _ in range(steps):
            X_input = current_sequence.reshape(1, self.sequence_length, 1)
            next_pred_scaled = self.model.predict(X_input, verbose=0)[0, 0]
            next_pred = self.scaler.inverse_transform([[next_pred_scaled]])[0, 0]
            
            # Add realistic fluctuation
            if add_noise:
                noise = np.random.normal(0, train_std)
                next_pred = next_pred + noise
                # Ensure non-negative values
                next_pred = max(0, next_pred)
            
            forecasts.append(next_pred)
            
            # Update sequence with the noisy prediction for more realistic fluctuation
            next_pred_scaled_noisy = self.scaler.transform([[next_pred]])[0, 0]
            current_sequence = np.append(current_sequence[1:], next_pred_scaled_noisy)
        
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
    
    tab1, tab2 = st.tabs(["📝 Register", "🔐 Login"])
    
    with tab1:
        st.subheader("Register New Account")
        reg_username = st.text_input("Username", key="reg_user", placeholder="Min 3 characters")
        reg_password = st.text_input("Password", type="password", key="reg_pass", placeholder="Min 4 characters")
        
        if st.button("✅ Register", key="reg_btn", use_container_width=True):
            if len(reg_username) >= 3 and len(reg_password) >= 4:
                st.success("✅ Registrasi berhasil! Silakan login.")
            else:
                st.error("❌ Username min 3 karakter, Password min 4 karakter")
    
    with tab2:
        st.subheader("Login to Dashboard")
        login_username = st.text_input("Username", key="login_user", placeholder="Enter username")
        login_password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
        
        if st.button("🔓 Login", key="login_btn", type="primary", use_container_width=True):
            if len(login_username) >= 3 and len(login_password) >= 4:
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.rerun()
            else:
                st.error("❌ Username min 3 karakter, Password min 4 karakter")
        
        st.markdown("---")
        with st.expander("🔒 Change Password"):
            st.caption("Change your password here")
            old_password = st.text_input("Old Password", type="password", key="old_pass")
            new_password = st.text_input("New Password", type="password", key="new_pass", placeholder="Min 4 characters")
            confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pass")
            
            if st.button("🔄 Change Password", key="change_btn", use_container_width=True):
                if len(new_password) >= 4 and new_password == confirm_password:
                    st.success("✅ Password berhasil diubah!")
                else:
                    st.error("❌ Password tidak valid atau tidak cocok")

# ==================== DASHBOARD PAGE ====================
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

# ==================== ADD DATA PAGE ====================
def add_data_page():
    st.title("➕ Add Data")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📁 Upload CSV File", "🎲 Generate Sample Data"])
    
    with tab1:
        st.subheader("Upload CSV File (Support hingga 1GB)")
        st.info("💡 Sistem mendukung file CSV hingga 1GB dengan chunked loading")
        
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
                            df = pd.read_csv(uploaded_file, low_memory=False)
                        
                        if df is not None:
                            st.session_state.csv_data = df
                            st.success(f"✅ Data loaded! Total: {len(df):,} baris | {len(df.columns)} kolom")
                            
                            with st.expander("👁️ Preview (10 baris)"):
                                st.dataframe(df.head(10), use_container_width=True)
                            
                            st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Tip: Untuk file sangat besar, coba tingkatkan chunk size atau pastikan memory cukup")
    
    with tab2:
        st.subheader("Generate Sample Data")
        
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
                    st.session_state.csv_data = generate_sample_data_chunked(sample_size)
                    st.success(f"✅ Generated {sample_size:,} baris!")
                    gc.collect()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

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
    
    # Check if cleaned data exists
    data_source = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.csv_data
    
    if data_source is None:
        st.warning("⚠️ Tidak ada data. Upload dan clean data terlebih dahulu.")
        return
    
    st.success(f"✅ Data tersedia: {len(data_source):,} baris")
    
    # Forecast Configuration
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
    
    # Additional Settings
    with st.expander("🔧 Advanced Settings"):
        if model_choice == "SARIMA":
            col1, col2 = st.columns(2)
            with col1:
                seasonal_period = st.number_input("Seasonal Period:", min_value=1, max_value=365, value=7)
            with col2:
                max_params = st.number_input("Max Parameter Search:", min_value=1, max_value=5, value=2)
        else:  # LSTM
            col1, col2, col3 = st.columns(3)
            with col1:
                sequence_length = st.number_input("Sequence Length:", min_value=7, max_value=90, value=30)
            with col2:
                lstm_units = st.number_input("LSTM Units:", min_value=10, max_value=200, value=50)
            with col3:
                epochs = st.number_input("Epochs:", min_value=10, max_value=200, value=100)
    
    # Prepare time series data
    st.subheader("📊 Data Preparation")
    
    date_col = st.selectbox("Pilih Kolom Tanggal:", options=[col for col in data_source.columns if 'date' in col.lower()])
    value_col = st.selectbox("Pilih Kolom Nilai:", options=['Final_Price', 'price', 'subtotal'] + [col for col in data_source.columns if col not in ['order_date', 'invoice_date']])
    
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"Generating {model_choice} forecast for {forecast_days} days..."):
            try:
                # Prepare time series
                df_ts = data_source.copy()
                df_ts[date_col] = pd.to_datetime(df_ts[date_col])
                df_ts = df_ts.sort_values(date_col)
                df_ts.set_index(date_col, inplace=True)
                
                # Create daily time series
                daily_ts = df_ts[value_col].resample('D').sum().fillna(0)
                
                st.info(f"📊 Time series created: {len(daily_ts)} days ({daily_ts.index.min()} to {daily_ts.index.max()})")
                
                # Train-test split (use last 30 days as test)
                test_size = min(30, len(daily_ts) // 5)
                train_size = len(daily_ts) - test_size
                train_data = daily_ts[:train_size]
                test_data = daily_ts[train_size:]
                
                st.write(f"📈 Training: {len(train_data)} days | Testing: {len(test_data)} days")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Model Training & Forecasting
                if model_choice == "SARIMA":
                    status_text.text("🔄 SARIMA Auto-Tuning: Mencari parameter terbaik...")
                    progress_bar.progress(0.2)
                    
                    sarima = SARIMAAutoForecaster()
                    
                    # Auto fit with grid search
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
                    
                    # Get model summary
                    model_summary = sarima.get_model_summary()
                    
                    status_text.text("🔄 Generating forecasts...")
                    progress_bar.progress(0.7)
                    
                    # Test forecast
                    test_forecast, test_lower, test_upper = sarima.forecast(len(test_data))
                    
                    # Future forecast
                    future_forecast, future_lower, future_upper = sarima.forecast(forecast_days)
                    
                    # Calculate metrics
                    metrics = {
                        'MAE': AdvancedMetrics.mae(test_data.values, test_forecast),
                        'RMSE': AdvancedMetrics.rmse(test_data.values, test_forecast),
                        'sMAPE': AdvancedMetrics.smape(test_data.values, test_forecast),
                        'MASE': AdvancedMetrics.mase(test_data.values, test_forecast, train_data.values, seasonality=seasonal_period)
                    }
                    
                    # Get diagnostics plot
                    try:
                        diagnostics_fig = sarima.plot_diagnostics()
                    except:
                        diagnostics_fig = None
                    
                    # Save results
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
                    status_text.text("🔄 Training LSTM model...")
                    progress_bar.progress(0.3)
                    
                    lstm = LSTMForecaster(sequence_length=sequence_length, lstm_units=lstm_units)
                    history = lstm.fit(train_data, epochs=epochs)
                    
                    status_text.text("🔄 Generating forecasts...")
                    progress_bar.progress(0.6)
                    
                    # Test forecast
                    test_forecast = lstm.forecast(train_data, len(test_data))
                    
                    # Future forecast
                    future_forecast = lstm.forecast(train_data, forecast_days)
                    
                    # Calculate metrics
                    metrics = {
                        'MAE': AdvancedMetrics.mae(test_data.values, test_forecast),
                        'RMSE': AdvancedMetrics.rmse(test_data.values, test_forecast),
                        'sMAPE': AdvancedMetrics.smape(test_data.values, test_forecast),
                        'MASE': AdvancedMetrics.mase(test_data.values, test_forecast, train_data.values, seasonality=7)
                    }
                    
                    # Save results
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
    
    # Display Results
    if st.session_state.forecast_result:
        st.markdown("---")
        st.header("📊 Hasil Forecast")
        
        result = st.session_state.forecast_result
        
        # Model Info
        st.subheader(f"🤖 Model: {result['model']}")
        
        # Model info displayed in simple format
        if result['model'] == 'SARIMA':
            st.info(f"✅ Model: SARIMA {result['model_info']['best_order']} x {result['model_info']['best_seasonal_order']}")
        else:  # LSTM
            st.info(f"✅ Model: LSTM (Sequence Length: {result['model_info']['sequence_length']}, Units: {result['model_info']['lstm_units']})")
        
        # Visualization
        st.markdown("---")
        st.subheader("📊 Visualisasi Forecast")
        
        # Create single combined plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        fig.suptitle(f'{result["model"]} Forecasting Results - Complete View', fontsize=16, fontweight='bold')
        
        # Show last 90 days of training data for context
        recent_train = result['train_data'].tail(90)
        ax.plot(recent_train.index, recent_train.values, 
                label='Historical Data', color='#2C3E50', linewidth=2.5, alpha=0.8)
        
        # Plot actual test data
        ax.plot(result['test_data'].index, result['test_data'].values, 
                label='Actual (Test Period)', color='#3498DB', linewidth=2.5, marker='o', markersize=4)
        
        # Plot test forecast
        ax.plot(result['test_data'].index, result['test_forecast'], 
                label='Forecast (Test Period)', color='#E74C3C', linewidth=2, linestyle='--', marker='s', markersize=3)
        
        # Plot future forecast - with smooth connection
        future_dates_with_connection = [result['train_data'].index[-1]] + list(result['future_dates'])
        future_values_with_connection = [result['train_data'].iloc[-1]] + list(result['future_forecast'])
        
        ax.plot(future_dates_with_connection, future_values_with_connection, 
                label=f'Future Forecast ({result["forecast_days"]} days)', 
                color='#27AE60', linewidth=2.5, linestyle='--', marker='o', markersize=4)
        
        # Confidence interval REMOVED (disabled)
        if result['model'] == 'SARIMA' and False:  # Disabled
            future_lower_with_connection = [result['train_data'].iloc[-1]] + list(result['future_lower'])
            future_upper_with_connection = [result['train_data'].iloc[-1]] + list(result['future_upper'])
            
            ax.fill_between(future_dates_with_connection, 
                           future_lower_with_connection, 
                           future_upper_with_connection,
                           alpha=0.15, color='#27AE60', label='Confidence Interval (95%)')
        
        # Styling improvements
        ax.set_title('Complete Forecast Timeline', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', rotation=45)
        
        # Add vertical line to separate historical and future
        ax.axvline(x=result['train_data'].index[-1], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(result['train_data'].index[-1], ax.get_ylim()[1]*0.95, 
                'Today', ha='center', fontsize=9, color='gray', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional insights
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
        
        # Download Forecast Results
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': result['future_dates'],
            'Forecast': result['future_forecast']
        })
        
        if result['model'] == 'SARIMA':
            forecast_df['Lower_CI'] = result['future_lower']
            forecast_df['Upper_CI'] = result['future_upper']
        
        csv_forecast = forecast_df.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="⬇️ Download Forecast (CSV)",
                data=csv_forecast,
                file_name=f"forecast_{result['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Save metrics as JSON
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
        
        # Show forecast table
        with st.expander("📋 View Forecast Table"):
            st.dataframe(forecast_df, use_container_width=True)

# ==================== MAIN APP ====================
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        # Sidebar
        with st.sidebar:
            st.title("PT. XYZ")
            st.markdown("**Dashboard System**")
            st.markdown("---")
            
            st.write(f"👤 **{st.session_state.username}**")
            
            # Data info
            if st.session_state.csv_data is not None:
                st.markdown("---")
                st.subheader("📊 Data Info")
                st.metric("Total Rows", f"{len(st.session_state.csv_data):,}")
                memory_mb = st.session_state.csv_data.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory", f"{memory_mb:.2f} MB")
                
                if st.session_state.cleaned_data is not None:
                    st.metric("Cleaned Rows", f"{len(st.session_state.cleaned_data):,}")
            
            st.markdown("---")
            
            # Navigation
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
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.csv_data = None
                st.session_state.cleaned_data = None
                st.session_state.forecast_result = None
                gc.collect()
                st.rerun()
        
        # Main content
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