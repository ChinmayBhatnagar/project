import os
import gpt_2_simple as gpt2
from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

# Download and load GPT-2 model (small version, 124M)
model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    gpt2.download_gpt2(model_name=model_name)  # model is saved into the current directory under /models/124M/

# Start a TensorFlow session and graph
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, model_name=model_name)

@app.route('/', methods=['GET', 'POST'])
def home():
    story = ""
    if request.method == 'POST':
        character = request.form['character']
        setting = request.form['setting']
        theme = request.form['theme']
        prompt = f"{character} in {setting} with a theme of {theme}"
        
        # Ensure the correct graph is used
        with sess.graph.as_default():
            story = gpt2.generate(sess, model_name=model_name, prefix=prompt, return_as_list=True, length=150)[0]
        
    return render_template('index.html', story=story)

if __name__ == '__main__':
    app.run(debug=True)
