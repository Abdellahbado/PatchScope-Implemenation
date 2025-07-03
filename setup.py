#!/usr/bin/env python3
"""
Setup script for PatchScope implementation.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_installation():
    """Check if required packages are available."""
    print("🔍 Checking installation...")
    
    packages = ["torch", "transformers", "accelerate"]
    missing = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return len(missing) == 0

def create_example_script():
    """Create an example script for first-time users."""
    example_content = '''#!/usr/bin/env python3
"""
Simple example script to get started with PatchScope.
"""

# Run this after setup to test your installation
from main import run_quick_test

if __name__ == "__main__":
    print("🚀 Running PatchScope quick test...")
    try:
        run_quick_test()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a GPU available or modify config.py for CPU usage.")
'''
    
    with open("example.py", "w") as f:
        f.write(example_content)
    
    print("📝 Created example.py script")

def main():
    """Main setup function."""
    print("🔧 PATCHSCOPE SETUP")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not os.path.exists("config.py"):
        print("❌ Please run this script from the PatchScope project directory")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed - could not install requirements")
        return
    
    # Check installation
    if not check_installation():
        print("❌ Setup incomplete - some packages missing")
        return
    
    # Create example script
    create_example_script()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run: python demo.py (to see the modular structure)")
    print("2. Run: python main.py (for quick test)")
    print("3. Run: python main.py --mode comprehensive (for full study)")
    print("4. Edit config.py to customize prompts and parameters")

if __name__ == "__main__":
    main()
