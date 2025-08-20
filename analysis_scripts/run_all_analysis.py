#!/usr/bin/env python3
"""
Master Analysis Script
Runs all visualization and analysis scripts for UCF101 CNN-RNN project
"""

import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """Run a Python script and handle any errors."""
    print(f"\n{'='*60}")
    print(f"🚀 Running: {script_name}")
    print(f"📝 Description: {description}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        print(f"✅ {script_name} completed successfully!")
        if result.stdout:
            print("📤 Output:", result.stdout.strip())
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(f"   Return code: {e.returncode}")
        if e.stdout:
            print(f"   Stdout: {e.stdout}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return False
    
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error running {script_name}: {e}")
        return False
    
    return True

def main():
    """Run all analysis scripts."""
    print("🎬 UCF101 CNN-RNN Project - Complete Analysis Suite")
    print("=" * 60)
    
    # List of scripts to run
    scripts = [
        ("plot_training_curves.py", "Training accuracy and loss curves"),
        ("plot_performance_distribution.py", "Performance distribution across all 101 classes"),
        ("plot_category_performance.py", "Performance breakdown by action categories"),
        ("plot_overfitting_analysis.py", "Overfitting analysis and gap detection")
    ]
    
    # Track success/failure
    successful = 0
    failed = 0
    
    # Run each script
    for script_name, description in scripts:
        if run_script(script_name, description):
            successful += 1
        else:
            failed += 1
        
        # Small delay between scripts
        time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 ANALYSIS COMPLETE - SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total: {len(scripts)}")
    
    if successful == len(scripts):
        print("\n🎉 All analysis scripts completed successfully!")
        print("📈 Generated visualizations:")
        print("   • training_curves.png - Training progress")
        print("   • performance_distribution.png - Class performance")
        print("   • category_performance.png - Category breakdown")
        print("   • overfitting_analysis.png - Overfitting analysis")
    else:
        print(f"\n⚠️  {failed} script(s) failed. Check the output above for details.")
    
    print(f"\n📁 All plots saved in current directory")
    print("🔍 Check the generated PNG files for detailed analysis")

if __name__ == "__main__":
    main() 