import os
from app import app

if __name__ == "__main__":
    # Use environment variable for port with fallback to 5000 (Flask's default port)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
