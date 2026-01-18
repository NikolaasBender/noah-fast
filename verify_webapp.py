import time
import requests
import sys

def verify():
    print("Waiting for server...")
    url = "http://localhost:8080"
    for i in range(10):
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                print("Server is UP and serving index.html!")
                if "AI Race Simulator" in resp.text:
                    print("Content verified.")
                    return
        except:
            pass
        time.sleep(2)
        print(f"Retry {i+1}...")
    
    print("Server failed to respond.")
    sys.exit(1)

if __name__ == "__main__":
    verify()
