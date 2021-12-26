from flask import Flask, request, jsonify
from chat_service import myChatBot
from flask.views import MethodView
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>API is Live and running</p>"

class ChatAPI(MethodView):
    def get(self):
        return "<h1>You have entered the Matrix</h1>"
    def post(self):
        content_type = request.headers.get('Content-Type')
        isAuthorized = request.headers.get('Secret-Key') == "Chicken"
        if not isAuthorized:
            return jsonify({'error': "Not Authorized Noob"}), 401
        if (content_type == 'application/json'):
            message = str(request.json['message'])
            response = myChatBot.getResponse(message)
            return jsonify({"botResponse": response}), 200
        else:
            return jsonify({'error': "Bad Request"}), 400
app.add_url_rule('/chatbot', view_func=ChatAPI.as_view('ChatAPI'))