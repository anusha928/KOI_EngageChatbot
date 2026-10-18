Part 1: Run the chatbot API

You need Python 3.10 or newer and a free Groq account.

1. Get a Groq API key

Sign in at https://console.groq.com, open API Keys and create one. Copy it somewhere safe. Never paste it into a file in this project or commit it to Git.

2. Create a virtual environment and install packages

bash
cd Engage_Chatbot
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Set your key as an environment variable

bash
export GROQ_API_KEY="your-key-here"            # Mac / Linux
# Windows PowerShell:  $env:GROQ_API_KEY="your-key-here"

The variable only lasts for that terminal window, so set it again in a new one.

4. Start the API

bash
python3 app.py

The first run downloads the embedding model (about 90 MB) and then caches it. When you see Running on http://127.0.0.1:5001 it is ready. The terminal prints which Groq model it picked, for example Using model: openai/gpt-oss-120b.

5. Test it (in a second terminal)

bash
curl http://127.0.0.1:5001/health
curl -X POST http://127.0.0.1:5001/chat -H "Content-Type: application/json" -d '{"message": "hi"}'

You should get a JSON reply with a reply and a confidence score.

Part 2: Put the chat on WordPress
A. Make the API reachable from the website

A WordPress page runs in the student's browser, so it cannot reach localhost. It needs a public HTTPS address.

For testing: ngrok

bash
brew install ngrok                # or download from ngrok.com
ngrok http 5001

Copy the https://....ngrok-free.dev address it prints. Your API address is that URL plus /chat. The address changes every time ngrok restarts, and the bot is only available while your computer and ngrok are running, so this is for testing only.

For real use: host app.py on an always-on service (a free tier host such as Render or Railway, or a KOI server). Use a production server such as gunicorn app:app instead of the built-in Flask server. This has not been tested yet. Check the host's memory limit first, because the embedding model needs a fair amount.

B. Upload the avatar
In WordPress go to Media > Add New and upload assets/koko.webp.
Click the image and copy its File URL.