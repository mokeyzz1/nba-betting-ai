# send_discord_daily.py
import subprocess
import os

python_path = "/Users/mk/nba-betting-ai/nba-env/bin/python"
discord_dir = "/Users/mk/nba-betting-ai/discord"

friends_script = os.path.join(discord_dir, "send_picks_to_friends.py")
private_script = os.path.join(discord_dir, "send_to_me.py")

# Run scripts (delay if needed in LaunchAgent)
subprocess.run([python_path, friends_script])
subprocess.run([python_path, private_script])
