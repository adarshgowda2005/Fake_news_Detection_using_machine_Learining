from flask import Flask, render_template, request, jsonify
import requests
import json
import re
import logging
from newspaper import Article

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class FakeNewsDetector:
    def __init__(self, ollama_url="http://localhost:11434/api/generate"):
        self.ollama_url = ollama_url
        self.model_name = "phi3.5:latest"
    
    def extract_article_content(self, url):
        """Extract main content from news URL"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            return {
                'title': article.title,
                'text': article.text[:2000],  # Limit text length
                'authors': article.authors,
                'publish_date': str(article.publish_date),
                'success': True
            }
        except Exception as e:
            logging.error(f"Error extracting article: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_news(self, text, url=None):
        """Analyze news content using Ollama"""
        
        system_prompt = """You are a professional fact-checking AI. Analyze the provided news content and determine if it's likely REAL or FAKE.

CRITERIA FOR ANALYSIS:
- Check for sensationalist language and emotional manipulation
- Look for verifiable facts vs opinions
- Assess the credibility based on writing style
- Identify logical inconsistencies
- Look for exaggerated claims or missing context

RESPONSE FORMAT:
You MUST respond with ONLY ONE WORD: either "REAL" or "FAKE" followed by a percentage confidence score.

Examples:
REAL 87%
FAKE 92%
REAL 76%

Do not include any explanations, additional text, or formatting. Just the one word and percentage."""

        user_prompt = f"""
NEWS CONTENT TO ANALYZE:
{text}

URL: {url if url else 'Not provided'}

Analysis:"""
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent output
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', '').strip()
                print(f"Ollama raw response: {analysis}")  # Debug line
                
                # Parse the response to extract REAL/FAKE and confidence
                return self._parse_response(analysis)
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")  # Debug line
                return {"error": f"Ollama API error: {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection error: {str(e)}"}
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}
    
    def _parse_response(self, response_text):
        """Parse the AI response to extract result and confidence"""
        # Clean the response text
        clean_text = response_text.strip()
        
        # Look for REAL or FAKE followed by optional percentage
        real_match = re.search(r'\bREAL\b', clean_text, re.IGNORECASE)
        fake_match = re.search(r'\bFAKE\b', clean_text, re.IGNORECASE)
        
        # Extract percentage (0-100)
        percentage_match = re.search(r'(\d{1,3})%', clean_text)
        confidence = int(percentage_match.group(1)) if percentage_match else 50
        
        if real_match:
            return {
                "verdict": "REAL",
                "confidence": confidence,
                "raw_response": clean_text
            }
        elif fake_match:
            return {
                "verdict": "FAKE", 
                "confidence": confidence,
                "raw_response": clean_text
            }
        else:
            # Fallback: try to determine from context
            if any(word in clean_text.lower() for word in ['real', 'true', 'accurate', 'credible']):
                return {"verdict": "REAL", "confidence": 60, "raw_response": clean_text}
            elif any(word in clean_text.lower() for word in ['fake', 'false', 'inaccurate', 'unreliable']):
                return {"verdict": "FAKE", "confidence": 60, "raw_response": clean_text}
            else:
                return {"verdict": "UNCERTAIN", "confidence": 50, "raw_response": clean_text}

# Initialize detector
detector = FakeNewsDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        text = data.get('text', '').strip()
        
        if not url and not text:
            return jsonify({'error': 'Please provide either a URL or text to analyze'})
        
        # If URL provided, extract content
        article_data = None
        if url:
            article_data = detector.extract_article_content(url)
            if not article_data['success']:
                return jsonify({'error': f'Failed to extract content from URL: {article_data["error"]}'})
            text = article_data['text']
        
        # Analyze the content
        result = detector.analyze_news(text, url)
        
        if 'error' in result:
            return jsonify({'error': result['error']})
        
        response_data = {
            'verdict': result['verdict'],
            'confidence': result['confidence'],
            'analysis_complete': True
        }
        
        # Add article info if available
        if article_data and article_data['success']:
            response_data['article_info'] = {
                'title': article_data['title'],
                'source': url
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'})

@app.route('/health')
def health_check():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return jsonify({'status': 'healthy', 'ollama_running': response.status_code == 200})
    except:
        return jsonify({'status': 'unhealthy', 'ollama_running': False})

@app.route('/debug')
def debug():
    """Debug endpoint to check Ollama connection"""
    try:
        # Test Ollama connection
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json() if response.status_code == 200 else None
        
        return jsonify({
            'ollama_status': response.status_code,
            'available_models': models,
            'selected_model': 'phi3.5:latest'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)