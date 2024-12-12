#!/bin/bash

# Update system and install dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3-pip \
    nginx \
    tesseract-ocr \
    ffmpeg \
    git

# Create app directory
sudo mkdir -p /opt/llm-search
sudo chown -R $USER:$USER /opt/llm-search

# Create virtual environment
python3.9 -m venv /opt/llm-search/venv
source /opt/llm-search/venv/bin/activate

# Clone repository (replace with your repo URL)
git clone https://your-repository-url.git /opt/llm-search/app

# Install requirements
pip install -r /opt/llm-search/app/requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/llm-search.service << EOF
[Unit]
Description=LLM Search Backend
After=network.target

[Service]
User=$USER
Group=$USER
WorkingDirectory=/opt/llm-search/app
Environment="PATH=/opt/llm-search/venv/bin"
ExecStart=/opt/llm-search/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx
sudo tee /etc/nginx/sites-available/llm-search << EOF
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/llm-search /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Start services
sudo systemctl daemon-reload
sudo systemctl start llm-search
sudo systemctl enable llm-search
sudo systemctl restart nginx 