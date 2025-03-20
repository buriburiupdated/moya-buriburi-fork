from flask import Flask, request, jsonify, render_template
from moya.tools.text_completion_azure import TextAutocomplete  # Import your class

app = Flask(__name__)
autocomplete = TextAutocomplete()

# Serve the index.html file
@app.route('/')
def home():
    return render_template('index.html')  # Flask automatically looks inside 'templates/' folder

@app.route('/complete', methods=['POST'])
def complete():
    data = request.json
    text = data.get('text', '')
    completion = autocomplete.complete_text(text)
    return jsonify({"completion": completion})

if __name__ == '__main__':
    app.run(debug=True)
