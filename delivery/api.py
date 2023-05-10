from flask import request, Flask, flash, Response
import json
import os
from werkzeug.utils import secure_filename
from application.application import Application
class Api:
    def __init__(self):
        app = Application()
    def predict(self):
        if request.method == "POST":
            if "obj" not in request.files:
                return Response(json.dumps({"error":" do not have obj file "}),status=400,mimetype="application/json")
            obj = request.files["obj"]
            if obj.filename == "":
                return Response(json.dumps({"error":" do not have obj file "}),status=400,mimetype="application/json")
            obj_name = secure_filename(obj.filename)
            path = os.getcwd()
            self.path = os.path.join("./reconstruction_file",obj_name)
            
