#!/bin/bash
# =============================================================================
# FastMCP Blockchain Analysis - Setup Script
# =============================================================================

echo "🚀 FastMCP Blockchain Analysis - Setup"
echo "======================================"

# Colors 
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}🐍 Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python 3.8+
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${GREEN}✅ Python version is compatible${NC}"
else
    echo -e "${RED}❌ Python 3.8+ required${NC}"
    exit 1
fi

# Create virtual environment (optional)
echo -e "${BLUE}🏗️  Creating virtual environment (optional)...${NC}"
read -p "Create virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    python3 -m venv fastmcp_env
    source fastmcp_env/bin/activate
    echo -e "${GREEN}✅ Virtual environment created and activated${NC}"
fi

# Install dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi

# Create directories
echo -e "${BLUE}📁 Creating project directories...${NC}"
mkdir -p data/hf_models_cache
mkdir -p data/api_cache
mkdir -p examples
mkdir -p logs

echo -e "${GREEN}✅ Directories created${NC}"

# Setup environment variables
echo -e "${BLUE}🔧 Setting up environment variables...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env file with your API keys:${NC}"
    echo "   - ALCHEMY_API_KEY (Required)"
    echo "   - HUGGINGFACE_TOKEN (Optional)"
    echo ""
    