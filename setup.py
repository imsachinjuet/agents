"""
Setup script for Supabase + LangChain integration
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = ".env"
    if not os.path.exists(env_path):
        print("❌ .env file not found!")
        return False
    
    required_vars = ["SUPABASE_URL", "SUPABASE_API_KEY"]
    missing_vars = []
    
    with open(env_path, 'r') as f:
        content = f.read()
        for var in required_vars:
            if var not in content or f"{var}=your_" in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing or incomplete environment variables: {missing_vars}")
        print("Please update your .env file with the correct values.")
        return False
    
    print("✅ Environment variables configured!")
    return True

def print_setup_instructions():
    """Print setup instructions"""
    print("\n" + "="*60)
    print("🎉 SUPABASE + LANGCHAIN SETUP COMPLETE!")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("1. Update your OpenAI API key in the .env file")
    print("2. (Optional) Add your Supabase database password to SUPABASE_DB_URI for SQL features")
    print("3. In your Supabase dashboard, create tables as needed")
    print("4. For vector search, create a documents table with vector column (see example file)")
    print("\n🔧 AVAILABLE FILES:")
    print("• agent1.py - Main agent with Supabase integration")
    print("• supabase_langchain_example.py - Vector store and RAG examples")
    print("• .env - Environment configuration (update with your keys)")
    print("\n🚀 TO RUN:")
    print("python agent1.py")
    print("\n📚 FOR VECTOR SEARCH DEMO:")
    print("python supabase_langchain_example.py")

def main():
    """Main setup function"""
    print("🚀 Setting up Supabase + LangChain integration...")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check environment configuration
    check_env_file()
    
    # Print final instructions
    print_setup_instructions()
    
    return True

if __name__ == "__main__":
    main()
