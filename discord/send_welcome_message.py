# -*- coding: utf-8 -*-
import requests

WEBHOOK_URL = "https://discord.com/api/webhooks/1355435204986015794/VPGghLuIPiDUaQC_mCAxh20Gk0rVQZU0XvDxJv6o_9ZAUFx03vue6EuRys3UD5py0Wmr"

def send_welcome_message():
    message = """
🏀 **Welcome to the NBA AI Picks Server!** 🤖

This server is powered by a custom-built **NBA prediction model** that posts daily picks using machine learning and real-time NBA data.

Each prediction includes:
• 🧠 Predicted Winner  
• 📊 Win Probability  
• 👍 Value Bet Flag

---

📈 **How It Works**  
The model uses team stats, recent form, effective FG%, betting odds, and implied win probabilities to make informed predictions using **XGBoost**.

Every day:
1. Pull games + odds  
2. Fetch advanced stats  
3. Make predictions  
4. Post results here automatically

---

⚠️ **Disclaimer:**  
These are purely AI-driven predictions. Use them for fun, discussion, or insights — but **always bet at your own risk**! 🎲  
No guarantees, no pressure. 

Thanks for joining! 🏀🔥
"""
    response = requests.post(WEBHOOK_URL, json={"content": message})

    if response.status_code == 204:
        print("✅ Welcome message sent!")
    else:
        print("❌ Failed to send message:", response.status_code, response.text)

# Run the function
send_welcome_message()
