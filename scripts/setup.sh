#!/bin/bash

# Job Analytics Project Setup Script
# This script sets up the development environment for the job analytics project

echo "🚀 Setting up Job Analytics Project..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p tests

# Set up pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
fi

# Create .env file from template
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp config.env.example .env
    echo "⚠️  Please update .env file with your configuration"
fi

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "🔧 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Job Analytics Project Setup"
fi

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Update .env file with your configuration"
echo "3. Run Jupyter: jupyter lab"
echo "4. Start dashboard: streamlit run dashboard/app.py"
echo "5. Start API: uvicorn api.main:app --reload"
echo ""
echo "🐳 Or use Docker: docker-compose up -d"
echo ""
echo "Happy coding! 🚀"
