from flask import request, Flask, flash, Response, send_from_directory
import json
import os
from werkzeug.utils import secure_filename
from application.application import Application
class Api:
    def __init__(self):
        self.app = Application()
    def predict(self):
        if request.method == "POST":
            if "obj" not in request.files:
                return Response(json.dumps({"error":" do not have obj file "}),status=400,mimetype="application/json")
            obj = request.files["obj"]
            if obj.filename == "":
                return Response(json.dumps({"error":" do not have obj file "}),status=400,mimetype="application/json")
            obj_name = secure_filename(obj.filename)
            self.path = os.path.join("./reconstruction_file",obj_name)
            obj.save(self.path)
            obj_json = dict()
            obj_json["reconstruction"] = self.app.execute_predict(self.path)
            obj_json["split"] = self.app.split_face()
            return Response(json.dumps(obj_json))
    def get_reconstruction_file(self):
        if request.method =="GET":
            if "file" in request.args:
                filename = request.args["file"]
                if not filename.endswith(".obj"):
                    return
                if not os.path.exists(os.path.join("./reconstructed_face_file",filename)):
                    return     
                return send_from_directory("./reconstructed_face_file", filename)
    def get_split_file(self):
        if request.method == "GET":
            if "file" in request.args:
                filename = request.args["file"]
                if not filename.endswith(".obj"):
                    return
                if not os.path.exists(os.path.join("./split_wound",filename)):
                    return
                return send_from_directory("./split_wound",filename)
