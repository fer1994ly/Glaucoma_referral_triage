import os
import logging
from app import app

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Get port from environment with fallback to 5000 (Replit's default external port)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting Flask server on port {port}")
    
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
            logger.error(f"Port {port} is already in use. Please ensure no other service is using this port.")
            # Attempt to kill any process using the port
            os.system(f"fuser -k {port}/tcp")
            logger.info(f"Attempting to restart on port {port}")
            app.run(host='0.0.0.0', port=port)
        else:
            logger.error(f"Failed to bind to port {port}: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
