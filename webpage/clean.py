from flask import Flask, render_template, request
app = Flask(__name__, template_folder='template')

from wtforms import (Form, validators, SubmitField, 
DecimalField, IntegerField, StringField)

class ReusableForm(Form):
    """User entry form for entering specifics for generation"""
    # Number of words
    words = StringField('Enter the sentence:', validators=[validators.DataRequired()])
    # Submit button
    submit = SubmitField("Enter")
    
# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)

    # Send template information to index.html
    return render_template('index.html', form=form)
app.run(host='0.0.0.0', port=50000)