from flask import Flask, request, jsonify, session, redirect, url_for, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging
from typing import Optional
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CSV_PATH = os.path.join(DATA_DIR, 'cgwb_tables.csv')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production-2024')
# Enable CORS for all routes with credentials - allow all origins for development
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# Initialize models and encoders
models = {}
encoders = {}

# Simple user database (in production, use a proper database)
# Default credentials: username='admin', password='admin123'
# Generate hashes once and store them
_admin_hash = generate_password_hash('admin123')
_user_hash = generate_password_hash('user123')

USERS = {
    'admin': _admin_hash,
    'user': _user_hash
}

# Verify hashes work on startup
logger.info("Verifying password hashes...")
assert check_password_hash(_admin_hash, 'admin123'), "Admin password hash verification failed"
assert check_password_hash(_user_hash, 'user123'), "User password hash verification failed"
logger.info("Password hashes verified successfully")

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def load_cgwb_data() -> Optional[pd.DataFrame]:
    """Attempt to load parsed CGWB CSV and coerce to a modeling dataframe.
    The PDF tables are heterogeneous; this function extracts any water level-like columns if present.
    Returns a dataframe with expected columns or None if not enough data.
    """
    try:
        if not os.path.isfile(CSV_PATH):
            return None
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return None
        # Create modeled features with best-effort mapping; many columns may not exist.
        # We synthesize geospatial and environmental features due to PDF limitations.
        n = len(df)
        rng = np.random.default_rng(42)
        modeled = pd.DataFrame({
            'latitude': rng.uniform(6.0, 37.0, n),
            'longitude': rng.uniform(68.0, 97.0, n),
            'soil_type': rng.choice(['sandy', 'clay', 'loam', 'silt'], n),
            'lithology': rng.choice(['granite', 'basalt', 'limestone', 'shale'], n),
            'land_use': rng.choice(['agriculture', 'forest', 'urban', 'grassland'], n),
            'rainfall_mm': rng.uniform(200, 2000, n),
            'slope_deg': rng.uniform(0, 30, n),
            'elevation_m': rng.uniform(0, 3000, n),
            'water_table_m': rng.uniform(1, 50, n),
            'distance_to_river_km': rng.uniform(0.1, 20, n),
            'ndvi': rng.uniform(0, 1, n)
        })
        # If any likely depth/water level columns exist, use them to adjust targets
        level_cols = [c for c in df.columns if 'water level' in c.lower() or 'wl' in c.lower() or 'depth' in c.lower()]
        if level_cols:
            levels = pd.to_numeric(df[level_cols[0]], errors='coerce')
            modeled['water_table_m'] = np.where(levels.notna(), np.clip(levels.values, 1, 60), modeled['water_table_m'])
        return modeled
    except Exception as ex:
        logger.warning(f"Failed to load/shape CGWB data: {ex}")
        return None


def create_sample_data():
    """Create sample training data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    data = {
        'latitude': np.random.uniform(6.0, 37.0, n_samples),  # India latitude range
        'longitude': np.random.uniform(68.0, 97.0, n_samples),  # India longitude range
        'soil_type': np.random.choice(['sandy', 'clay', 'loam', 'silt'], n_samples),
        'lithology': np.random.choice(['granite', 'basalt', 'limestone', 'shale'], n_samples),
        'land_use': np.random.choice(['agriculture', 'forest', 'urban', 'grassland'], n_samples),
        'rainfall_mm': np.random.uniform(200, 2000, n_samples),
        'slope_deg': np.random.uniform(0, 30, n_samples),
        'elevation_m': np.random.uniform(0, 3000, n_samples),
        'water_table_m': np.random.uniform(1, 50, n_samples),
        'distance_to_river_km': np.random.uniform(0.1, 20, n_samples),
        'ndvi': np.random.uniform(0, 1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variables based on realistic relationships
    # Suitability: combination of factors
    suitability_score = (
        (df['rainfall_mm'] > 800) * 0.3 +
        (df['soil_type'].isin(['loam', 'sandy'])) * 0.2 +
        (df['lithology'].isin(['granite', 'basalt'])) * 0.2 +
        (df['water_table_m'] < 20) * 0.2 +
        (df['distance_to_river_km'] < 5) * 0.1
    )
    
    df['suitable'] = (suitability_score > 0.5).astype(int)
    
    # Depth prediction (deeper wells in dry areas, shallow in wet areas)
    df['depth_m'] = np.random.uniform(10, 100, n_samples)
    df.loc[df['rainfall_mm'] < 500, 'depth_m'] += 20
    df.loc[df['water_table_m'] > 30, 'depth_m'] += 15
    
    # Discharge prediction (higher in suitable areas)
    df['discharge_lps'] = np.random.uniform(0.5, 10, n_samples)
    df.loc[df['suitable'] == 1, 'discharge_lps'] *= 1.5
    
    # Quality index (0-100)
    df['quality_index'] = np.random.uniform(60, 95, n_samples)
    df.loc[df['suitable'] == 0, 'quality_index'] -= 20
    
    return df

def train_models():
    """Train machine learning models"""
    logger.info("Creating and training models...")
    # Prefer CGWB-parsed data if available; otherwise synthetic
    df = load_cgwb_data()
    if df is None or df.empty:
        logger.info("Using synthetic dataset (CGWB parsed data unavailable or empty)")
        df = create_sample_data()
    
    # Prepare features
    categorical_features = ['soil_type', 'lithology', 'land_use']
    numerical_features = ['latitude', 'longitude', 'rainfall_mm', 'slope_deg', 
                         'elevation_m', 'water_table_m', 'distance_to_river_km', 'ndvi']
    
    # Encode categorical variables
    for feature in categorical_features:
        le = LabelEncoder()
        df[f'{feature}_encoded'] = le.fit_transform(df[feature])
        encoders[feature] = le
    
    # Prepare feature matrix
    feature_columns = [f'{f}_encoded' for f in categorical_features] + numerical_features
    X = df[feature_columns]
    
    # Train suitability classifier
    y_suitable = df['suitable']
    models['suitability'] = RandomForestClassifier(n_estimators=100, random_state=42)
    models['suitability'].fit(X, y_suitable)
    
    # Train depth regressor
    y_depth = df['depth_m']
    models['depth'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['depth'].fit(X, y_depth)
    
    # Train discharge regressor
    y_discharge = df['discharge_lps']
    models['discharge'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['discharge'].fit(X, y_discharge)
    
    # Train quality regressor
    y_quality = df['quality_index']
    models['quality'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['quality'].fit(X, y_quality)
    
    logger.info("Models trained successfully!")

def preprocess_input(data):
    """Preprocess input data for prediction"""
    # Create a copy of the input data
    processed = data.copy()
    
    # Encode categorical variables
    categorical_features = ['soil_type', 'lithology', 'land_use']
    for feature in categorical_features:
        if feature in processed and feature in encoders:
            try:
                processed[f'{feature}_encoded'] = encoders[feature].transform([processed[feature]])[0]
            except ValueError:
                # Handle unseen categories
                processed[f'{feature}_encoded'] = 0
    
    # Prepare feature vector in the same order as training
    feature_columns = [f'{f}_encoded' for f in categorical_features] + [
        'latitude', 'longitude', 'rainfall_mm', 'slope_deg', 
        'elevation_m', 'water_table_m', 'distance_to_river_km', 'ndvi'
    ]
    
    feature_vector = []
    for col in feature_columns:
        if col in processed:
            feature_vector.append(processed[col])
        else:
            feature_vector.append(0)  # Default value for missing features
    
    return np.array(feature_vector).reshape(1, -1)

def get_readymade_result(data: dict) -> dict:
    """Return predefined, India-centric results when ML results aren't available.
    Values are indicative and tailored per land_use type.
    """
    land_use = str(data.get('land_use', '')).strip().lower()
    presets = {
        'agriculture': {
            'suitable_probability': 0.78,
            'predicted_depth_m': 40.0,
            'predicted_discharge_lps': 6.0,
            'predicted_quality_index': 82.0,
        },
        'forest': {
            'suitable_probability': 0.72,
            'predicted_depth_m': 35.0,
            'predicted_discharge_lps': 5.5,
            'predicted_quality_index': 85.0,
        },
        'urban': {
            'suitable_probability': 0.42,
            'predicted_depth_m': 65.0,
            'predicted_discharge_lps': 3.0,
            'predicted_quality_index': 70.0,
        },
        'grassland': {
            'suitable_probability': 0.55,
            'predicted_depth_m': 50.0,
            'predicted_discharge_lps': 4.0,
            'predicted_quality_index': 75.0,
        }
    }

    preset = presets.get(land_use, {
        'suitable_probability': 0.5,
        'predicted_depth_m': 50.0,
        'predicted_discharge_lps': 4.0,
        'predicted_quality_index': 75.0,
    })

    suitable_flag = 1 if preset['suitable_probability'] >= 0.5 else 0
    return {
        'suitable': suitable_flag,
        'suitable_probability': float(preset['suitable_probability']),
        'predicted_depth_m': float(preset['predicted_depth_m']),
        'predicted_discharge_lps': float(preset['predicted_discharge_lps']),
        'predicted_quality_index': float(preset['predicted_quality_index']),
        'input_data': data,
        'source': 'readymade'
    }

@app.route('/login', methods=['POST'])
def login():
    """Login endpoint"""
    try:
        data = request.get_json()
        if not data:
            logger.warning("Login attempt with no data")
            return jsonify({'error': 'Invalid request. Please provide username and password.'}), 400
            
        username = data.get('username', '').strip().lower()
        password = data.get('password', '').strip()
        
        if not username or not password:
            logger.warning(f"Login attempt with missing credentials for user: {username}")
            return jsonify({'error': 'Username and password are required'}), 400
        
        # Lookup user (username is already lowercased)
        if username not in USERS:
            logger.warning(f"Login attempt with unknown username: '{username}'")
            logger.info(f"Available usernames: {list(USERS.keys())}")
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Debug: Log password check (without logging actual password)
        stored_hash = USERS[username]
        logger.debug(f"Attempting login for user: {username}, password length: {len(password)}")
        password_match = check_password_hash(stored_hash, password)
        logger.info(f"Password check for {username}: {password_match}")
        
        if password_match:
            session['logged_in'] = True
            session['username'] = username
            logger.info(f"User {username} logged in successfully")
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'username': username
            })
        else:
            logger.warning(f"Login attempt with incorrect password for user: {username}")
            # For debugging: verify the hash is valid
            try:
                test_hash = generate_password_hash('test')
                test_check = check_password_hash(test_hash, 'test')
                logger.info(f"Password hash system test: {test_check}")
            except Exception as hash_test_error:
                logger.error(f"Password hash system error: {hash_test_error}")
            return jsonify({'error': 'Invalid username or password'}), 401
            
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/logout', methods=['POST'])
def logout():
    """Logout endpoint"""
    session.clear()
    logger.info("User logged out")
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/check-auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if 'logged_in' in session and session['logged_in']:
        return jsonify({
            'authenticated': True,
            'username': session.get('username', '')
        })
    return jsonify({'authenticated': False}), 401

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Main prediction endpoint"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    # Check authentication
    if 'logged_in' not in session or not session['logged_in']:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")
        
        # Validate required fields
        required_fields = ['soil_type', 'lithology', 'land_use', 'latitude', 'longitude']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Set default values for optional fields
        defaults = {
            'rainfall_mm': 800,
            'slope_deg': 5.0,
            'elevation_m': 400,
            'water_table_m': 15.0,
            'distance_to_river_km': 2.0,
            'ndvi': 0.4
        }
        
        for key, default_value in defaults.items():
            if key not in data or data[key] is None:
                data[key] = default_value
        
        # Static/readymade override or missing models fallback
        if bool(data.get('use_static')) or not models or any(k not in models for k in ['suitability', 'depth', 'discharge', 'quality']):
            result = get_readymade_result(data)
            logger.info(f"Returning readymade result: {result}")
            return jsonify(result)

        # Preprocess input and run ML models
        X = preprocess_input(data)

        suitability_pred = models['suitability'].predict(X)[0]
        suitability_prob = models['suitability'].predict_proba(X)[0][1]
        depth_pred = models['depth'].predict(X)[0]
        discharge_pred = models['discharge'].predict(X)[0]
        quality_pred = models['quality'].predict(X)[0]

        result = {
            'suitable': int(suitability_pred),
            'suitable_probability': float(suitability_prob),
            'predicted_depth_m': float(depth_pred),
            'predicted_discharge_lps': float(discharge_pred),
            'predicted_quality_index': float(quality_pred),
            'input_data': data,
            'source': 'model'
        }

        logger.info(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        try:
            data = request.get_json() or {}
            # Ensure we have minimum required data for fallback
            if not data.get('land_use'):
                data['land_use'] = 'agriculture'
            result = get_readymade_result(data)
            result['warning'] = 'Using fallback prediction due to model error'
            logger.info(f"Returning readymade fallback result: {result}")
            return jsonify(result)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {str(fallback_error)}")
            return jsonify({
                'error': 'Prediction failed',
                'message': str(e),
                'suitable': 0,
                'suitable_probability': 0.0,
                'predicted_depth_m': 0.0,
                'predicted_discharge_lps': 0.0,
                'predicted_quality_index': 0.0
            }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0,
        'available_models': list(models.keys())
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint - serve homepage"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        home_path = os.path.join(base_dir, 'home.html')
        with open(home_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving home.html: {e}")
        # Fallback to login page if home.html doesn't exist
        try:
            login_path = os.path.join(base_dir, 'login.html')
            with open(login_path, 'r', encoding='utf-8') as f:
                return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
        except:
            return f"Error loading home page: {str(e)}", 500

@app.route('/home.html', methods=['GET'])
def home_page():
    """Serve homepage"""
    return home()

@app.route('/about.html', methods=['GET'])
def about_page():
    """Serve about page"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        about_path = os.path.join(base_dir, 'about.html')
        with open(about_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving about.html: {e}")
        return f"Error loading about page: {str(e)}", 500

@app.route('/features.html', methods=['GET'])
def features_page():
    """Serve features page"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        features_path = os.path.join(base_dir, 'features.html')
        with open(features_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving features.html: {e}")
        return f"Error loading features page: {str(e)}", 500

@app.route('/login.html', methods=['GET'])
def login_page():
    """Serve login page"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        login_path = os.path.join(base_dir, 'login.html')
        with open(login_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving login.html: {e}")
        return f"Error loading login page: {str(e)}", 500

@app.route('/index.html', methods=['GET'])
def index():
    """Serve main application page"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(base_dir, 'index.html')
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return f"Error loading index page: {str(e)}", 500

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'message': 'AI Water Well Predictor API',
        'version': '1.0.0',
        'endpoints': {
            'POST /login': 'User login',
            'POST /logout': 'User logout',
            'GET /check-auth': 'Check authentication status',
            'POST /predict': 'Make water well predictions (requires auth)',
            'GET /health': 'Health check',
            'GET /api': 'This information'
        },
        'required_fields': ['soil_type', 'lithology', 'land_use', 'latitude', 'longitude'],
        'optional_fields': ['rainfall_mm', 'slope_deg', 'elevation_m', 'water_table_m', 'distance_to_river_km', 'ndvi']
    })

if __name__ == '__main__':
    # Train models on startup
    train_models()
    
    # Run the app
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)

