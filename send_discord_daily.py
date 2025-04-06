# send_discord_daily.py

import subprocess
import os

python_path = "/Users/mk/nba-betting-ai/nba-env/bin/python"
discord_dir = "/Users/mk/nba-betting-ai/discord"

friends_script = os.path.join(discord_dir, "send_to_friends.py")
private_script = os.path.join(discord_dir, "send_to_me.py")

print("📢 Sending Discord update to friends...")
try:
    subprocess.run([python_path, friends_script], check=True)
except Exception as e:
    print(f"❌ Failed to send to friends: {e}")

print("🔒 Sending Discord update to YOU...")
try:
    subprocess.run([python_path, private_script], check=True)
except Exception as e:
    print(f"❌ Failed to send to yourself: {e}")
