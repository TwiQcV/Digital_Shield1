# 🔒 Ḥimā - Digital Shield Chatbot

A professional Streamlit chatbot application for cybersecurity threat intelligence and digital protection guidance.

## Features

### 🎨 **Centered Avatar Design**
- Centered avatar image from `Digital_Shield_Avatars/Welcome.jpg`
- Professional branding with Saudi heritage theme
- Clean, modern UI with cybersecurity color scheme

### 💬 **Interactive Chat Interface**
- Real-time chat with message history
- User messages displayed on the right
- Assistant responses on the left
- Professional styling with code blocks for technical content

### 🛡️ **Cybersecurity Intelligence**
- Phishing protection guidance
- Ransomware trends and prevention
- Banking security best practices
- General cybersecurity advice

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure RAG System (Optional but Recommended)
For AI-powered responses, ensure your RAG system is properly configured:

```bash
# Check if .env file exists with GEMINI_API_KEY
ls -la .env

# If not, create one with your Google Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Ensure your cybersecurity dataset is available
ls -la Digital_Shield_data/processed/
```

### 3. Run the Chatbot
```bash
# Option 1: Using the runner script
python run_chatbot.py

# Option 2: Direct Streamlit command
streamlit run streamlit_chatbot.py
```

### 4. Access the Application
- The application will open automatically in your browser
- Default URL: `http://localhost:8501`

## Application Structure

### Top Section (Centered)
```
[Avatar Image - Centered]
🔒 Ḥimā - Securing Your Digital World
With Saudi heritage of protection

💡 Quick Questions:
• Phishing protection?
• Ransomware trends?
• Bank security?
```

### Main Chat Area (Full Width)
```
💬 Ḥimā Cyber Threat Intelligence Chat
Ḥimā is here to protect your digital world
---
[Chat messages area]
[Assistant: responses on left]
[User: messages on right]
🔍 Ask about cyber threats... [input field]
```

## Key Features

### 🤖 **AI-Powered Intelligence**
The chatbot now uses advanced RAG (Retrieval-Augmented Generation) technology with Google Gemini AI:

**Primary Mode - RAG System:**
- **Intelligent Analysis**: Powered by Google Gemini AI
- **Real-time Data**: Access to comprehensive cybersecurity databases
- **Contextual Responses**: Based on actual threat intelligence from your Digital Shield dataset
- **Smart Suggestions**: Related questions and follow-ups
- **Document Analysis**: Responses based on analysis of multiple cybersecurity documents

**Fallback Mode - Hardcoded Responses:**
- **Phishing Protection**: Detection techniques, common tactics, protection strategies, mobile protection, banking industry specifics
- **Ransomware**: Current threat landscape, prevention strategies, response plans, banking industry compliance
- **Banking Security**: Account protection, transaction security, network infrastructure, compliance monitoring, employee training
- **Global Cyber Threats**: Most affected regions, threat actor origins, banking sector specifics, emerging trends, regional defense strategies
- **General Security**: Core security principles, specific topics, quick tips, tailored guidance

### 💾 **Session Management**
- Chat history maintained during the session
- Messages persist until browser refresh
- Clean conversation flow

### 🔍 **RAG System Status**
The chatbot displays its current mode:
- **🤖 AI-Powered with RAG System**: When RAG system is active and ready
- **💬 Standard Mode**: When using fallback responses
- **System Info**: Shows document count and analysis details in responses

### 🎨 **Professional Styling**
- Cybersecurity-themed color scheme (#1f4e79, #2c5aa0)
- Responsive design
- Professional typography
- Gradient headers and clean containers

## File Structure

```
├── streamlit_chatbot.py      # Main Streamlit application
├── run_chatbot.py           # Easy launcher script
├── requirements.txt         # Dependencies (updated with Streamlit)
├── Digital_Shield_Avatars/
│   └── Welcome.jpg         # Avatar image
└── CHATBOT_README.md       # This documentation
```

## Customization

### Adding New Response Types
Edit the `generate_response()` function in `streamlit_chatbot.py` to add new cybersecurity topics or modify existing responses.

### Styling Changes
Modify the CSS in the `st.markdown()` section at the top of the application to customize colors, fonts, and layout.

### Avatar Image
Replace `Digital_Shield_Avatars/Welcome.jpg` with your preferred avatar image. The application will automatically detect and display it.

## Technical Details

### Dependencies
- `streamlit>=1.28.0` - Web application framework
- `pathlib` - File path handling
- `os` - Operating system interface

### Session State
- Uses Streamlit's `st.session_state` for chat history
- Messages stored as list of dictionaries with `role` and `content`

### Responsive Design
- Uses Streamlit columns and containers for layout
- CSS styling for professional appearance
- Mobile-friendly responsive design

## Security Features

The chatbot emphasizes:
- **Multi-factor authentication**
- **Regular software updates**
- **Strong password practices**
- **Security awareness training**
- **Incident response planning**

## Support

For issues or questions about the Ḥimā chatbot:
1. Check that all dependencies are installed
2. Verify the avatar image exists in the correct path
3. Ensure Streamlit is properly installed
4. Check browser console for any JavaScript errors

---

**🔒 Stay Secure with Ḥimā - Your Digital Shield!**
