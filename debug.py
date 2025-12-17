import pymysql
import pandas as pd
from datetime import datetime

# ============== DATABASE CONNECTION ==============
def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='pt_xyz_test1',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )

# ============== STEP 1: CHECK TABLE STRUCTURE ==============
def check_table_structure():
    """Check if upload_history table exists and has correct structure"""
    print("=" * 80)
    print("STEP 1: CHECKING TABLE STRUCTURE")
    print("=" * 80)
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Check if table exists
            cursor.execute("SHOW TABLES LIKE 'upload_history'")
            result = cursor.fetchone()
            
            if not result:
                print("❌ ERROR: Table 'upload_history' TIDAK DITEMUKAN!")
                print("\n💡 SOLUSI: Jalankan SQL berikut untuk membuat tabel:\n")
                print("""
CREATE TABLE IF NOT EXISTS upload_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    username VARCHAR(100) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_size_mb DECIMAL(10,2),
    total_rows INT,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'success',
    error_message TEXT,
    INDEX idx_user_id (user_id),
    INDEX idx_upload_date (upload_date),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
                return False
            
            # Table exists, check structure
            print("✅ Table 'upload_history' ditemukan!")
            cursor.execute("DESCRIBE upload_history")
            columns = cursor.fetchall()
            
            print("\n📋 Struktur Tabel:")
            print("-" * 80)
            for col in columns:
                print(f"  {col['Field']:<20} {col['Type']:<20} {col['Null']:<10} {col['Key']}")
            
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        conn.close()

# ============== STEP 2: CHECK DATA ==============
def check_data_content():
    """Check if there's any data in upload_history"""
    print("\n" + "=" * 80)
    print("STEP 2: CHECKING DATA CONTENT")
    print("=" * 80)
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Count total records
            cursor.execute("SELECT COUNT(*) as total FROM upload_history")
            total = cursor.fetchone()['total']
            
            print(f"\n📊 Total Records: {total}")
            
            if total == 0:
                print("⚠️  WARNING: Tidak ada data dalam tabel upload_history!")
                print("\n💡 KEMUNGKINAN PENYEBAB:")
                print("   1. Fungsi log_upload_history() gagal menyimpan data")
                print("   2. User belum pernah upload CSV sejak tabel dibuat")
                print("   3. Ada error saat upload yang tidak tertangkap")
                return False
            
            # Show latest 10 records
            cursor.execute("""
                SELECT 
                    id, user_id, username, filename, 
                    file_size_mb, total_rows, upload_date, status
                FROM upload_history 
                ORDER BY upload_date DESC 
                LIMIT 10
            """)
            records = cursor.fetchall()
            
            print("\n📋 10 Latest Records:")
            print("-" * 80)
            df = pd.DataFrame(records)
            print(df.to_string(index=False))
            
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        conn.close()

# ============== STEP 3: CHECK QUERY YANG DIGUNAKAN ==============
def test_admin_query():
    """Test query yang digunakan di admin dashboard"""
    print("\n" + "=" * 80)
    print("STEP 3: TESTING ADMIN QUERY")
    print("=" * 80)
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Query exact seperti di code
            query = """
                SELECT 
                    u.username,
                    uh.filename,
                    uh.file_size_mb,
                    uh.total_rows,
                    uh.upload_date,
                    uh.status,
                    uh.error_message
                FROM upload_history uh
                JOIN users u ON uh.user_id = u.id
                WHERE u.is_admin = 0
                ORDER BY uh.upload_date DESC
                LIMIT 100
            """
            
            print("\n🔍 Query yang digunakan:")
            print("-" * 80)
            print(query)
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            print(f"\n📊 Result Count: {len(results)}")
            
            if len(results) == 0:
                print("\n⚠️  WARNING: Query tidak mengembalikan hasil!")
                print("\n💡 KEMUNGKINAN PENYEBAB:")
                print("   1. Semua user adalah admin (is_admin = 1)")
                print("   2. Tidak ada relasi user_id yang cocok")
                print("   3. Data upload_history kosong")
                
                # Check users
                cursor.execute("SELECT id, username, is_admin FROM users")
                users = cursor.fetchall()
                print("\n👥 User List:")
                print("-" * 80)
                for user in users:
                    admin_status = "ADMIN" if user['is_admin'] == 1 else "USER"
                    print(f"  ID: {user['id']}, Username: {user['username']}, Status: {admin_status}")
                
                return False
            
            print("\n📋 Query Results:")
            print("-" * 80)
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        print("\n🔍 Full Traceback:")
        print(traceback.format_exc())
        return False
    finally:
        conn.close()

# ============== STEP 4: INSERT TEST DATA ==============
def insert_test_data():
    """Insert test data untuk testing"""
    print("\n" + "=" * 80)
    print("STEP 4: INSERT TEST DATA")
    print("=" * 80)
    
    response = input("\n⚠️  Apakah Anda ingin insert test data? (yes/no): ")
    
    if response.lower() != 'yes':
        print("❌ Skipped")
        return
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Get first non-admin user
            cursor.execute("SELECT id, username FROM users WHERE is_admin = 0 LIMIT 1")
            user = cursor.fetchone()
            
            if not user:
                print("❌ ERROR: Tidak ada user non-admin!")
                print("💡 Buat user baru terlebih dahulu")
                return
            
            # Insert test data
            test_data = {
                'user_id': user['id'],
                'username': user['username'],
                'filename': 'test_upload.csv',
                'file_size_mb': 2.5,
                'total_rows': 10000,
                'status': 'success',
                'error_message': None
            }
            
            cursor.execute("""
                INSERT INTO upload_history 
                (user_id, username, filename, file_size_mb, total_rows, status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                test_data['user_id'],
                test_data['username'],
                test_data['filename'],
                test_data['file_size_mb'],
                test_data['total_rows'],
                test_data['status'],
                test_data['error_message']
            ))
        
        conn.commit()
        print(f"✅ Test data berhasil diinsert untuk user: {user['username']}")
        
        # Verify
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM upload_history ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            print("\n📋 Inserted Data:")
            print("-" * 80)
            for key, value in result.items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ ERROR: {e}")
        import traceback
        print("\n🔍 Full Traceback:")
        print(traceback.format_exc())
    finally:
        conn.close()

# ============== STEP 5: FIX CODE RECOMMENDATIONS ==============
def show_fix_recommendations():
    """Show code fix recommendations"""
    print("\n" + "=" * 80)
    print("STEP 5: CODE FIX RECOMMENDATIONS")
    print("=" * 80)
    
    print("""
📝 RECOMMENDED FIXES:

1. **Add Debug Logging to log_upload_history():**

def log_upload_history(user_id, username, filename, file_size_mb, total_rows, status='success', error_msg=None):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            query = '''
                INSERT INTO upload_history 
                (user_id, username, filename, file_size_mb, total_rows, status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            '''
            cursor.execute(query, (user_id, username, filename, file_size_mb, total_rows, status, error_msg))
        conn.commit()
        
        # ✅ ADD THIS: Verify insert
        with conn.cursor() as cursor:
            cursor.execute("SELECT LAST_INSERT_ID() as last_id")
            last_id = cursor.fetchone()['last_id']
            print(f"✅ Upload history logged with ID: {last_id}")
        
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ log_upload_history ERROR: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    finally:
        if conn:
            conn.close()

2. **Update get_latest_uploads_admin() with Better Error Handling:**

def get_latest_uploads_admin():
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            query = '''
                SELECT 
                    u.username,
                    uh.filename,
                    uh.file_size_mb,
                    uh.total_rows,
                    uh.upload_date,
                    uh.status,
                    uh.error_message
                FROM upload_history uh
                JOIN users u ON uh.user_id = u.id
                WHERE u.is_admin = 0
                ORDER BY uh.upload_date DESC
                LIMIT 100
            '''
            cursor.execute(query)
            results = cursor.fetchall()
            
            # ✅ ADD THIS: Debug print
            print(f"📊 Admin Query returned {len(results)} records")
        
        if results and len(results) > 0:
            return pd.DataFrame(results)
        else:
            print("⚠️ No upload history found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ get_latest_uploads_admin ERROR: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

3. **Force Upload History Tab to Show Debug Info:**

# In admin_dashboard_page(), add this in Upload History tab:

with tab2:
    st.subheader("📤 Recent CSV Uploads")
    
    # ✅ ADD THIS: Debug button
    if st.button("🔍 Debug Upload History"):
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Check total records
                cursor.execute("SELECT COUNT(*) as total FROM upload_history")
                total = cursor.fetchone()['total']
                st.info(f"Total records in database: {total}")
                
                # Check user filter
                cursor.execute('''
                    SELECT COUNT(*) as total 
                    FROM upload_history uh
                    JOIN users u ON uh.user_id = u.id
                    WHERE u.is_admin = 0
                ''')
                filtered_total = cursor.fetchone()['total']
                st.info(f"Records for non-admin users: {filtered_total}")
        except Exception as e:
            st.error(f"Debug error: {e}")
        finally:
            conn.close()
    
    # ... rest of your code ...
    """)

# ============== MAIN EXECUTION ==============
def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("🔍 PT. XYZ - UPLOAD HISTORY DEBUG SCRIPT")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run all checks
    table_ok = check_table_structure()
    
    if not table_ok:
        print("\n❌ CRITICAL: Table structure issue detected!")
        print("💡 Please create the upload_history table first")
        return
    
    data_ok = check_data_content()
    query_ok = test_admin_query()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Table Structure: {'✅ OK' if table_ok else '❌ FAILED'}")
    print(f"Data Content: {'✅ OK' if data_ok else '⚠️  EMPTY'}")
    print(f"Admin Query: {'✅ OK' if query_ok else '❌ FAILED'}")
    print("=" * 80)
    
    if not data_ok:
        insert_test_data()
    
    show_fix_recommendations()
    
    print("\n" + "=" * 80)
    print("✅ DIAGNOSTIC COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()