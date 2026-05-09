import subprocess
import sys
import os
import webbrowser

python_exe = sys.executable

app_path = os.path.join(os.path.dirname(__file__), "app.py")

subprocess.Popen([python_exe, app_path])

webbrowser.open("http://localhost:5000")

print("Spam Classifier is running at http://localhost:5000")
print("Close this terminal window to stop the server.")
