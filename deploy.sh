#!/bin/bash

# Deployment script for PosePython app on EC2
# Run this from your cloned repository directory

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.10 and pip
echo "Installing Python 3.10..."
sudo apt-get install -y python3.10 python3.10-venv python3-pip

# Install system dependencies for OpenCV
echo "Installing system dependencies..."
sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Install nginx
echo "Installing nginx..."
sudo apt-get install -y nginx

# Get current directory (should be your cloned repo)
APP_DIR=$(pwd)
echo "Using app directory: $APP_DIR"

# Ensure ubuntu user owns the directory
sudo chown -R ubuntu:ubuntu $APP_DIR

# Create Python virtual environment
echo "Creating virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create output directory
mkdir -p output_videos

# Set up nginx
echo "Configuring nginx..."
sudo cp nginx.conf /etc/nginx/sites-available/poseapp
sudo ln -s /etc/nginx/sites-available/poseapp /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/poseapp.service > /dev/null <<EOF
[Unit]
Description=Pose Detection Flask App
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/gunicorn app:app --bind 127.0.0.1:5000 --workers 4 --timeout 300 --worker-class sync
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
echo "Starting application service..."
sudo systemctl daemon-reload
sudo systemctl enable poseapp
sudo systemctl start poseapp

# Check service status
sudo systemctl status poseapp

echo "Deployment complete!"
echo "Your app should be accessible at http://cyclofit-ai.grity.co"
echo "To update your app:"
echo "  1. git pull"
echo "  2. sudo systemctl restart poseapp"
echo "Check logs with: sudo journalctl -u poseapp -f" 