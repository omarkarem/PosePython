# Complete AWS EC2 Deployment Guide for Pose Detection App

## Prerequisites
- AWS Account
- Domain/subdomain (cyclofit-ai.grity.co) pointing to your EC2 instance
- Basic knowledge of SSH and terminal commands

## Step 1: Launch EC2 Instance

1. **Log in to AWS Console** and navigate to EC2

2. **Launch Instance:**
   - Click "Launch Instance"
   - Name: "PoseApp-Server"
   - Select Ubuntu Server 22.04 LTS (HVM), SSD Volume Type
   - Instance type: t3.medium (minimum) or t3.large (recommended for better performance)
   - Key pair: Create new or use existing (download .pem file)
   - Network settings:
     - Allow SSH traffic from your IP
     - Allow HTTP traffic from anywhere (0.0.0.0/0)
     - Allow HTTPS traffic from anywhere (0.0.0.0/0)
   - Storage: 20 GB gp3 (SSD)
   - Click "Launch Instance"

3. **Note the Public IP address** of your instance

## Step 2: Configure Domain DNS

1. Go to your domain registrar (where you bought grity.co)
2. Add an A record:
   - Host: cyclofit-ai
   - Points to: [Your EC2 Public IP]
   - TTL: 300 seconds

## Step 3: Connect to EC2 Instance

```bash
# Make your key file secure
chmod 400 your-key-file.pem

# Connect via SSH
ssh -i your-key-file.pem ubuntu@[YOUR-EC2-PUBLIC-IP]
```

## Step 4: Clone Your Repository

```bash
# Once connected, run these commands:

# Update system
sudo apt update && sudo apt upgrade -y

# Install git
sudo apt install -y git

# Clone your repository
git clone https://github.com/yourusername/PosePython.git
# OR if private repo:
# git clone https://username:token@github.com/yourusername/PosePython.git

# Navigate to your repo
cd PosePython
```

## Step 5: Remove Unnecessary Files

```bash
# Remove files not needed for EC2 deployment
rm -f Procfile Aptfile vercel.json runtime.txt build.sh outputApp.py
```

## Step 6: Run Deployment Script

```bash
# Make deployment script executable
chmod +x deploy.sh

# Run the deployment script (from inside your repo directory)
./deploy.sh
```

## Step 7: SSL Certificate Setup (HTTPS)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d cyclofit-ai.grity.co

# Follow the prompts:
# - Enter email address
# - Agree to terms
# - Choose whether to redirect HTTP to HTTPS (recommended: yes)
```

## Step 8: Configure Firewall

```bash
# Set up UFW firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

## Step 9: Set Up Monitoring and Logs

```bash
# View application logs
sudo journalctl -u poseapp -f

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Monitor system resources
htop  # Install with: sudo apt install htop
```

## Step 10: Easy Updates (Main Benefit!)

**This is where the new approach shines:**

```bash
# To update your application:
cd ~/PosePython  # or wherever you cloned it
git pull
sudo systemctl restart poseapp

# That's it! Your app is updated.
```

## Step 11: Maintenance Commands

```bash
# Restart application
sudo systemctl restart poseapp

# Stop application
sudo systemctl stop poseapp

# Start application
sudo systemctl start poseapp

# Check status
sudo systemctl status poseapp

# Restart nginx
sudo systemctl restart nginx

# Check if your app is running on the correct port
sudo lsof -i :5000
```

## Step 12: Performance Optimization

1. **Edit nginx configuration** for better performance:
```bash
sudo nano /etc/nginx/sites-available/poseapp
```

Add these optimizations inside the server block:
```nginx
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_proxied expired no-cache no-store private auth;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/json application/xml;
    gzip_disable "MSIE [1-6]\.";
```

2. **Increase file upload limits** if needed:
```bash
sudo nano /etc/nginx/nginx.conf
```
Add in http block:
```nginx
client_max_body_size 100M;
```

3. **Restart nginx:**
```bash
sudo nginx -t
sudo systemctl restart nginx
```

## Step 13: Security Best Practices

1. **Set up automatic security updates:**
```bash
sudo apt install unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

2. **Regular backups:**
```bash
# Create backup script
cat > ~/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf ~/backups/poseapp_$DATE.tar.gz ~/PosePython
# Keep only last 7 days of backups
find ~/backups -name "poseapp_*.tar.gz" -mtime +7 -delete
EOF

chmod +x ~/backup.sh
mkdir -p ~/backups

# Add to crontab for daily backups
crontab -e
# Add this line:
# 0 2 * * * /home/ubuntu/backup.sh
```

## Troubleshooting

### If the app doesn't start:
```bash
# Check logs
sudo journalctl -u poseapp -n 50

# Check if port 5000 is in use
sudo lsof -i :5000

# Manually test the app
cd ~/PosePython
source venv/bin/activate
python app.py
```

### If nginx shows 502 Bad Gateway:
```bash
# Check if app is running
sudo systemctl status poseapp

# Check nginx error log
sudo tail -f /var/log/nginx/error.log

# Restart both services
sudo systemctl restart poseapp
sudo systemctl restart nginx
```

### If domain doesn't work:
```bash
# Check DNS propagation
nslookup cyclofit-ai.grity.co

# Check nginx configuration
sudo nginx -t
```

### If updates don't work:
```bash
# Make sure you're in the right directory
cd ~/PosePython
pwd  # Should show /home/ubuntu/PosePython

# Check git status
git status

# Force pull if needed
git reset --hard origin/main
git pull

# Restart app
sudo systemctl restart poseapp
```

## Deployment Workflow

**For future updates:**

1. **Develop locally** - make your changes
2. **Push to GitHub** - `git push`
3. **Update on server:**
   ```bash
   ssh -i your-key.pem ubuntu@[EC2-IP]
   cd ~/PosePython
   git pull
   sudo systemctl restart poseapp
   ```

## Cost Optimization

- Use t3.medium instance: ~$30/month
- Enable auto-stop during low usage
- Use CloudFlare for CDN (free tier)
- Monitor with AWS CloudWatch

## Final Notes

- Your app will be accessible at: https://cyclofit-ai.grity.co
- Easy updates with just `git pull` + `sudo systemctl restart poseapp`
- All your code stays in your GitHub repo location
- No need to copy files around

---

## Quick Reference Commands

```bash
# SSH to server
ssh -i your-key.pem ubuntu@[EC2-IP]

# Update and restart app
cd ~/PosePython && git pull && sudo systemctl restart poseapp

# View logs
sudo journalctl -u poseapp -f

# Check app status
sudo systemctl status poseapp
``` 