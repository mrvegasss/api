from flask import Flask, request, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Model ve tokenizer yükleme
model_name = "meta/code-llama-7b-hf"  # Çok dilli kodlama için uygun bir model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# HTML şablon (Render için kullanılacak)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Generation API</title>
</head>
<body>
    <h1>Code Generation Result</h1>
    <h3>Input Prompt:</h3>
    <pre>{{ prompt }}</pre>
    <h3>Generated Code:</h3>
    <pre>{{ generated_code }}</pre>
</body>
</html>
"""

@app.route('/generate', methods=['POST'])
def generate_code():
    try:
        # İstekten veriyi al
        data = request.json
        prompt = data.get("prompt", "")
        
        if not prompt:
            return {"error": "Prompt is required!"}, 400

        # Model girişini hazırla
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Kod üretimi
        outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Render edilmiş HTML döndür
        return render_template_string(HTML_TEMPLATE, prompt=prompt, generated_code=generated_code)
    
    except Exception as e:
        return {"error": str(e)}, 500

# API Test için bir anasayfa
@app.route('/', methods=['GET'])
def index():
    return '''
    <h1>Code Generation API</h1>
    <p>Send a POST request to <code>/generate</code> with a JSON body:</p>
    <pre>{
        "prompt": "def hello_world():\\n    print('Hello, World!')"
    }</pre>
    '''

if __name__ == '__main__':
    app.run(debug=True)
