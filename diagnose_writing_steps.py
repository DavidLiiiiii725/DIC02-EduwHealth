"""
Writing Steps Diagnostic Script
"""

import os

print("=" * 60)
print("Writing Steps Diagnostic")
print("=" * 60)

# 1. Check JS file
js_path = "tutor/static/tutor/js/writing-steps.js"
if os.path.exists(js_path):
    print(f"[OK] JS file exists: {js_path}")
    with open(js_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'function initStepCards' in content:
            print("  [OK] initStepCards function found")
        if 'function parseTaskSteps' in content:
            print("  [OK] parseTaskSteps function found")
        if 'function showStep' in content:
            print("  [OK] showStep function found")
        if 'function nextStep' in content:
            print("  [OK] nextStep function found")
        if 'function prevStep' in content:
            print("  [OK] prevStep function found")
else:
    print(f"[ERROR] JS file not found: {js_path}")

print()

# 2. Check HTML template
html_path = "tutor/templates/tutor/writing.html"
if os.path.exists(html_path):
    print(f"[OK] HTML template exists: {html_path}")
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        if 'writing-steps.js' in content:
            print("  [OK] JS file is included")
        else:
            print("  [ERROR] JS file not included")
            
        if 'step-card-container' in content:
            print("  [OK] Step card container exists")
        else:
            print("  [ERROR] Step card container missing")
            
        if 'initStepCards' in content:
            print("  [OK] initStepCards call exists")
        else:
            print("  [ERROR] initStepCards call missing")
            
        if 'id="step-content"' in content:
            print("  [OK] Step content area exists")
        else:
            print("  [ERROR] Step content area missing")
            
        if 'id="btn-next-step"' in content:
            print("  [OK] Next button exists")
        else:
            print("  [ERROR] Next button missing")
            
        if 'id="btn-prev-step"' in content:
            print("  [OK] Previous button exists")
        else:
            print("  [ERROR] Previous button missing")
else:
    print(f"[ERROR] HTML template not found: {html_path}")

print()
print("=" * 60)
print("Diagnostic complete!")
print()
print("If all checks passed, please:")
print("1. Restart Django server")
print("2. Clear browser cache (Ctrl+Shift+Delete)")
print("3. Open Writing page")
print("4. Click 'Generate Task'")
print("5. Check if step cards appear")
print()
print("If still not working, open browser console (F12) for errors")
print("=" * 60)
