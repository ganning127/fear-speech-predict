# Fear Speech Detection
84% accuracy using SVMs.

## Documentation
Base URL for all endpoints: https://fear-speech-predict.herokuapp.com

`GET` `predict/detect-fear`: predicts for fear spech using the SVM model.
- Send the query paramter `text` with the text that you want to predict for fear speech or not

## Local Setup
1. Run `source venv/bin/activate`
2. Run `venv/bin/activate`
3. Run `pip3 install -r requirements.txt`

> **Note**: If you get an Application Error, that means I've run out of free API time on Heroku 😅
