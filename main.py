import os
import signal
import sys
import logging
from app import app
from flask import jsonify

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
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Get port from environment with fallback to 3001 (Replit's standard port)
    port = int(os.environ.get('PORT', 3001))
    
    logger.info(f"Starting Flask server on port {port}")
    logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
    logger.info(f"Debug mode: {app.debug}")
    
    try:
        # Start the Flask application with production settings
        app.run(
            host='0.0.0.0',  # Listen on all available interfaces
            port=port,
            debug=False,  # Disable debug mode in production
            threaded=True,  # Enable threading for better performance
            use_reloader=False  # Disable reloader in production
        )
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is already in use. Process details:", exc_info=True)
            logger.error(f"Attempted port: {port}")
            # Log running processes on this port
            try:
                import subprocess
                result = subprocess.run(['lsof', f'-i:{port}'], capture_output=True, text=True)
                logger.error(f"Processes using port {port}:\n{result.stdout}")
            except Exception as proc_e:
                logger.error(f"Failed to get process information: {proc_e}")
            sys.exit(1)
        else:
            logger.error(f"Failed to bind to port {port}:", exc_info=True)
            sys.exit(1)
    except Exception as e:
        logger.error("Critical error during server startup:", exc_info=True)
        sys.exit(1)
