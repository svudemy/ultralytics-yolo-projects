import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===== GLOBAL CONFIGURATION (CAN BE MODIFIED AS NEEDED) =====

# Cooldown time in seconds to prevent spamming alerts
COOLDOWN = 6
# Variable to track the last time an alert was sent
LAST_SENT_TIME = 0 
# Telegram bot token
TELEGRAM_TOKEN = '8674565631:AAHN_DWSEG-_wuyF8mITJsn8rDX340yQi5o'  
# Telegram chat ID to send alerts to
RECEIVER_ID = 000000

# Distance threshold (in pixels) to consider an object as a potential weapon based on its proximity to the person
WEAPON_DISTANCE = 60