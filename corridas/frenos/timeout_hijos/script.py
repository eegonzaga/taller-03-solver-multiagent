import subprocess, sys, time
hijo = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
open('pid_hijo.txt', 'w').write(str(hijo.pid))
time.sleep(600)
