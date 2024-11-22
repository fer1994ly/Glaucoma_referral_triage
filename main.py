import os
import signal
import sys
import logging
from app import app
from flask import jsonify
import datetime

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
    # Don't exit immediately, let Flask handle the shutdown
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create persistent directories if they don't exist
        os.makedirs('data', exist_ok=True)  # For database
        os.makedirs('uploads', exist_ok=True)  # For uploaded files
        os.makedirs('logs', exist_ok=True)  # For log files
        
        port = int(os.getenv('PORT', 5000))
        
        # Ensuring the server runs continuously
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True  # Enable threading for better request handling
        )
    except Exception as e:
        logger.error("Critical error during server startup:", exc_info=True)
        sys.exit(1)