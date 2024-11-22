import os
import signal
import sys
import logging
from app import app  # Ensure app is imported from app.py
from flask import jsonify
import datetime  # Import datetime for timestamp

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()})

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}. Starting graceful shutdown...")
    sys.exit(0)

if __name__ == "__main__":
    try:
        # Ensuring the server runs continuously without terminating unexpectedly
        app.run(
            host='0.0.0.0',  # Allows all available network interfaces
            port=5000,       # Default port for Flask
            debug=False,     # Ensure debug mode is off for production
            use_reloader=False  # Disable reloader for production
        )
    except Exception as e:
        logger.error("Critical error during server startup:", exc_info=True)
        sys.exit(1)  # Exit to indicate we'll not handle the exception further