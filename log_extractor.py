import redis
import time
import os

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
OUTPUT_FILE = "global_log.txt"

# Povezivanje na Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def ensure_log_header():
    """Ako fajl ne postoji, dodaj zaglavlje."""
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("TIMESTAMP | NAME | NODE_ID | EVENT\n")
            f.write("-" * 60 + "\n")

def main():
    ensure_log_header()
    last_id = "0-0"  # počinjemo od najstarijeg loga
    print("[GlobalLogExtractor] Starting log extractor...")

    while True:
        try:
            # Čekamo nove logove max 5 sekundi
            entries = redis_client.xread({"global_logs": last_id}, block=5000, count=50)
            if entries:
                for stream, messages in entries:
                    for msg_id, fields in messages:
                        timestamp = fields.get(b"timestamp", b"").decode()
                        name = fields.get(b"name", b"").decode()
                        node_id = fields.get(b"node_id", b"").decode()
                        event = fields.get(b"event", b"").decode()

                        log_line = f"{timestamp} | {name} | {node_id} | {event}"
                        print(log_line)

                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                            f.write(log_line + "\n")

                        last_id = msg_id
        except Exception as e:
            print(f"[GlobalLogExtractor] Error: {e}")
            time.sleep(2)  # mali delay ako nešto pukne

if __name__ == "__main__":
    main()
