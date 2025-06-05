#!/bin/bash

# YOLO 11 Pose Detection Deployment Script for EC2
# This script sets up the YOLO pose detection app on Ubuntu EC2 instance

set -e  # Exit on any error

echo "🚀 Starting YOLO 11 Pose Detection Deployment"
echo "=============================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.10 and essential packages
echo "🐍 Installing Python 3.10..."
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev python3-pip -y

# Install system dependencies for OpenCV and YOLO
echo "📸 Installing OpenCV dependencies..."
sudo apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran

# Install FFmpeg for video processing
echo "🎥 Installing FFmpeg..."
sudo apt install ffmpeg -y

# Install nginx for reverse proxy
echo "🌐 Installing nginx..."
sudo apt install nginx -y

# Create virtual environment (optional - comment out if using system-wide)
# echo "🐍 Creating virtual environment..."
# python3.10 -m venv yolo_venv
# source yolo_venv/bin/activate

# Install Python packages
echo "📦 Installing Python packages..."
pip3 install --upgrade pip
pip3 install -r requirements_yolo.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p output_videos
mkdir -p yolo_training_data/{images/{train,val},labels/{train,val}}

# Download pre-trained YOLO model if not exists
echo "🎯 Downloading YOLO 11 pose model..."
if [ ! -f "models/cycling_pose_model.pt" ]; then
    python3 -c "from ultralytics import YOLO; model = YOLO('yolo11n-pose.pt'); model.save('models/cycling_pose_model.pt')"
    echo "✅ YOLO 11 pose model downloaded"
fi

# Create nginx configuration
echo "🌐 Configuring nginx..."
sudo tee /etc/nginx/sites-available/yolo-pose > /dev/null << EOF
server {
    listen 80;
    server_name yolo-pose.grity.co;  # Replace with your domain

    client_max_body_size 100M;
    client_body_timeout 300s;
    proxy_read_timeout 300s;
    
    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/yolo-pose /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Install PM2 for process management
echo "⚙️ Installing PM2..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g pm2

# Create PM2 ecosystem file
echo "⚙️ Creating PM2 configuration..."
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [
    {
      name: 'yolo-pose-app',
      script: 'python3',
      args: 'yolo_app.py',
      cwd: '$(pwd)',
      instances: 1,
      exec_mode: 'fork',
      max_memory_restart: '2G',
      env: {
        FLASK_ENV: 'production',
        PYTHONPATH: '$(pwd)'
      },
      error_file: './logs/yolo-pose-err.log',
      out_file: './logs/yolo-pose-out.log',
      log_file: './logs/yolo-pose-combined.log',
      time: true
    }
  ]
};
EOF

# Create logs directory
mkdir -p logs

# Start PM2 application
echo "🚀 Starting YOLO pose application..."
pm2 start ecosystem.config.js
pm2 save
pm2 startup

# Set up firewall
echo "🔥 Configuring firewall..."
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 5001/tcp    # Flask app (for testing)
sudo ufw --force enable

# Create useful aliases
echo "⚡ Creating useful aliases..."
cat >> ~/.bashrc << EOF

# YOLO Pose App aliases
alias yolo-status='pm2 status'
alias yolo-logs='pm2 logs yolo-pose-app'
alias yolo-restart='pm2 restart yolo-pose-app'
alias yolo-stop='pm2 stop yolo-pose-app'
alias yolo-monitor='pm2 monit'
alias train-yolo='python3 training.py'
EOF

source ~/.bashrc

echo ""
echo "🎉 YOLO 11 Pose Detection deployment completed!"
echo "=============================================="
echo ""
echo "📋 Service Information:"
echo "  • Application: http://$(curl -s ifconfig.me):5001"
echo "  • Local test: http://localhost:5001"
echo "  • Health check: http://localhost:5001/health"
echo ""
echo "🛠️ Management Commands:"
echo "  • Check status: pm2 status"
echo "  • View logs: pm2 logs yolo-pose-app"
echo "  • Restart app: pm2 restart yolo-pose-app"
echo "  • Train model: python3 training.py"
echo ""
echo "📁 Important directories:"
echo "  • Models: $(pwd)/models/"
echo "  • Training data: $(pwd)/yolo_training_data/"
echo "  • Output videos: $(pwd)/output_videos/"
echo ""
echo "🔗 Next steps:"
echo "  1. Point your domain (yolo-pose.grity.co) to this server's IP"
echo "  2. Set up SSL with: sudo certbot --nginx -d yolo-pose.grity.co"
echo "  3. Train custom model with: python3 training.py (if you have training data)"
echo "  4. Test the service at http://$(curl -s ifconfig.me):5001"
echo "" 