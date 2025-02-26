
[     UTC     ] Logs for tradingapp5-tmk4fdevugsqejy2pb6q6x.streamlit.app/
────────────────────────────────────────────────────────────────────────────────────────
[18:58:09] 🖥 Provisioning machine...
[18:58:09] 🎛 Preparing system...
[18:58:09] ⛓ Spinning up manager process...
[18:58:10] 🚀 Starting up repository: 'tradingapp5', branch: 'main', main module: 'app.py'
[18:58:10] 🐙 Cloning repository...
[18:58:10] 🐙 Cloning into '/mount/src/tradingapp5'...

[18:58:10] 🐙 Cloned repository!
[18:58:10] 🐙 Pulling code changes from Github...
[18:58:11] 📦 Processing dependencies...

──────────────────────────────────────── uv ───────────────────────────────────────────

Using uv pip install.
Using Python 3.12.9 environment at /home/adminuser/venv
Resolved 48 packages in 1.37s
Prepared [2025-02-26 18:58:19.130127] 48 packages[2025-02-26 18:58:19.130401]  [2025-02-26 18:58:19.130631] in 6.01s[2025-02-26 18:58:19.130884] 
Installed 48 packages in 741ms
 + altair==5.5.0
 + attrs==25.1.0
 + beautifulsoup4==4.12.3
 + blinker==1.9.0
 + cachetools==5.5.2
 + certifi==2025.1.31
 + charset-normalizer==3.4.1
 + click==[2025-02-26 18:58:19.874791] 8.1.8
 + feedparser==6.0.11
 + gitdb==4.0.12
 + gitpython==3.1.44
 + idna==3.10
 + jinja2==3.1.5
 + jsonschema==4.23.0
 + jsonschema-specifications==2024.10.1
 + lxml==5.2.1
 + markdown-it-py==3.0.0
 + markupsafe==3.0.2
 + mdurl==[2025-02-26 18:58:19.875023] 0.1.2
 + narwhals==1.28.0
 + numpy==1.26.4
 + packaging==24.2[2025-02-26 18:58:19.875231] 
 + pandas==2.2.1
 + pillow==[2025-02-26 18:58:19.875466] 10.4.0
 + plotly==5.18.0
 + protobuf==4.25.6
 + [2025-02-26 18:58:19.876010] pyarrow==19.0.1
 + pydeck==0.9.1
 + pygments==2.19.1[2025-02-26 18:58:19.876196] 
 + python-dateutil==2.9.0.post0
 + pytz==2024.1
 + [2025-02-26 18:58:19.876389] referencing==0.36.2
 + requests==2.31.0
 + rich==13.9.4
 + rpds-py==0.23.1
 + setuptools==70.0.0
 [2025-02-26 18:58:19.876761] + sgmllib3k==1.0.0
 + six==1.17.0
 + smmap==5.0.2[2025-02-26 18:58:19.876939] 
 + soupsieve==2.6
 + streamlit==1.35.0
 + [2025-02-26 18:58:19.877107] tenacity==8.5.0
 + toml==0.10.2
 + tornado==[2025-02-26 18:58:19.877278] 6.4.2
 + typing-extensions==4.12.2
 + tzdata==2025.1
 [2025-02-26 18:58:19.877472] + urllib3==2.3.0
 + watchdog==6.0.0
Checking if Streamlit is installed
Found Streamlit version 1.35.0 in the environment
Installing rich for an improved exception logging
Using uv pip install.
Using Python 3.12.9 environment at /home/adminuser/venv
Audited 1 package in 3ms

────────────────────────────────────────────────────────────────────────────────────────

[18:58:22] 🐍 Python dependencies were installed from /mount/src/tradingapp5/requirements.txt using uv.
Check if streamlit is installed
Streamlit is already installed
[18:58:24] 📦 Processed dependencies!



2025-02-26 19:03:16.031 503 GET /script-health-check (127.0.0.1) 2742.88ms
[19:03:16] ❗️ 
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:600 in _run_script                                      
                                                                                
  /mount/src/tradingapp5/app.py:115 in <module>                                 
                                                                                
    112 if __name__ == "__main__":                                              
    113 │   if not os.path.exists(CACHE_DIR):                                   
    114 │   │   os.makedirs(CACHE_DIR)                                          
  ❱ 115 │   main()                                                              
    116                                                                         
────────────────────────────────────────────────────────────────────────────────
NameError: name 'main' is not defined
2025-02-26 19:03:18.336 503 GET /script-health-check (127.0.0.1) 13.31ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:600 in _run_script                                      
                                                                                
  /mount/src/tradingapp5/app.py:115 in <module>                                 
                                                                                
    112 if __name__ == "__main__":                                              
    113 │   if not os.path.exists(CACHE_DIR):                                   
    114 │   │   os.makedirs(CACHE_DIR)                                          
  ❱ 115 │   main()                                                              
    116                                                                         
────────────────────────────────────────────────────────────────────────────────
NameError: name 'main' is not defined
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:600 in _run_script                                      
                                                                                
  /mount/src/tradingapp5/app.py:115 in <module>                                 
                                                                                
    112 if __name__ == "__main__":                                              
    113 │   if not os.path.exists(CACHE_DIR):                                   
    114 │   │   os.makedirs(CACHE_DIR)                                          
  ❱ 115 │   main()                                                              
    116                                                                         
────────────────────────────────────────────────────────────────────────────────
NameError: name 'main' is not defined
2025-02-26 19:03:23.382 503 GET /script-health-check (127.0.0.1) 12.95ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:600 in _run_script                                      
                                                                                
  /mount/src/tradingapp5/app.py:115 in <module>                                 
                                                                                
    112 if __name__ == "__main__":                                              
    113 │   if not os.path.exists(CACHE_DIR):                                   
    114 │   │   os.makedirs(CACHE_DIR)                                          
  ❱ 115 │   main()                                                              
    116                                                                         
────────────────────────────────────────────────────────────────────────────────
NameError: name 'main' is not defined
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:600 in _run_script                                      
                                                                                
  /mount/src/tradingapp5/app.py:115 in <module>                                 
                                                                                
    112 if __name__ == "__main__":                                              
    113 │   if not os.path.exists(CACHE_DIR):                                   
    114 │   │   os.makedirs(CACHE_DIR)                                          
  ❱ 115 │   main()                                                              
    116                                                                         
────────────────────────────────────────────────────────────────────────────────
NameError: name 'main' is not defined
2025-02-26 19:03:28.307 503 GET /script-health-check (127.0.0.1) 13.33ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:600 in _run_script                                      
                                                                                
  /mount/src/tradingapp5/app.py:115 in <module>                                 
                                                                                
    112 if __name__ == "__main__":                                              
    113 │   if not os.path.exists(CACHE_DIR):                                   
    114 │   │   os.makedirs(CACHE_DIR)                                          
  ❱ 115 │   main()                                                              
    116                                                                         
────────────────────────────────────────────────────────────────────────────────
NameError: name 'main' is not defined
2025-02-26 19:03:33.548 503 GET /script-health-check (127.0.0.1) 195.28ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:600 in _run_script                                      
                                                                                
  /mount/src/tradingapp5/app.py:115 in <module>                                 
                                                                                
    112 if __name__ == "__main__":                                              
    113 │   if not os.path.exists(CACHE_DIR):                                   
    114 │   │   os.makedirs(CACHE_DIR)                                          
  ❱ 115 │   main()                                                              
    116                                                                         
────────────────────────────────────────────────────────────────────────────────
NameError: name 'main' is not defined
2025-02-26 19:03:38.344 503 GET /script-health-check (127.0.0.1) 13.20ms
