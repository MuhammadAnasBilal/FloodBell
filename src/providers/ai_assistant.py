import requests
import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def generate_chat_response(user_message: str, context_data: dict) -> str:
    """
    General Climate Assistant using DeepSeek API.
    """
    
    # Base system prompt for General Climate Assistant
    system_prompt = f"""You are DeepSeek Climate AI, an expert general climate and weather assistant.
While you specialize in Pakistan's flood management, you can answer any general climate questions.
Current Map Location: {context_data.get('location')}
Flood Risk Level at Location: {context_data.get('risk_level')}
Current Weather: {context_data.get('weather')}
Local Hydrology (Rivers/Dams): {context_data.get('hydrology')}
Model Probability of Flood: {context_data.get('probability')}

Instructions:
- Provide clear, concise, and professional responses.
- If the user asks about evacuation, advise them based on the Flood Risk Level.
- Act as a comprehensive climate assistant.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=15)
        if response.status_code != 200:
            print(f"DeepSeek Error: {response.text}")
            return f"DeepSeek API Error: {response.status_code}. Please ensure you provided a valid DEEPSEEK_API_KEY in the environment."
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"DeepSeek API Exception: {e}")
        return f"Network error: Unable to reach the DeepSeek intelligence cluster. ({str(e)})"
