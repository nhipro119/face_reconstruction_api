from GraphAE import graphAE as graphAE
from GraphAE import graphAE_param as Param
from GraphAE import graphAE_dataloader as Dataloader
import trimesh
import numpy as np
import torch
import os
import subprocess
class Application:
    def __init__(self):
        self.param = Param.Parameters()
        self.param.batch =1
        self.param.read_config(".\\GraphAE\\0422_graphAE_dfaust\\10_conv_res.config")
        self.param.read_weight_path = ".\\GraphAE\\0422_graphAE_dfaust\\weight_10\\model_epoch0201.weight"
        self.model = graphAE.Model(param=self.param, test_mode=True)
        self.model.cpu()
        checkpoint = torch.load(self.param.read_weight_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.init_test_mode()
        self.model.eval()
        
        # ip = subprocess.check_output("wget -q -O- icanhazip.com",text=True, shell=True, stderr=subprocess.STDOUT).replace("\n","")
        # self.link = "http://"+ip+":5001"
        self.link = "a"
    def execute_predict(self, file_path):
        mesh = trimesh.exchange.load.load(file_path)
        ver = np.asarray(mesh.vertices)
        ver, height = self.preproccessing(ver)
        ver_out = self.predict(ver)
        ver_out = self.after_proccessing(ver_out, height)
        mesh = self.change_ver(mesh, ver_out)
        self.filename = file_path.split("/")[-1]
        save_file_path = self.save_predicted_obj(mesh,self.filename)
        download_link = self.link+"/reconstructed-face?file="+self.filename
        return download_link
    def save_predicted_obj(self, mesh, name):
        file_save_path = os.path.join("./reconstructed_face_file",name)
        mesh.export(file_save_path)
        return file_save_path
    def preproccessing(self, ver):
        ver = np.expand_dims(ver, axis=0)
        height = ver[:,:,1].mean(1)
        ver[:,:,0:3] -= ver[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(self.param.point_num,1)
        ver = torch.FloatTensor(ver).cpu()
        return ver, height
    def predict(self, ver):
        ver_out = self.model(ver)
        return ver_out
    def after_proccessing(self, ver_out, height):
        ver_out = np.array(ver_out[0].data.tolist())
        ver_out += height[0]
        return ver_out
    def change_ver(self, mesh, ver):
        mesh.vertices = ver
        return mesh
    def split_face(self):
        input_obj_path = os.path.join("./wound_face_file",self.filename)
        output_obj_path = os.path.join("./reconstruction_face_file", self.filename)
        obj_in = trimesh.load(input_obj_path)
        obj_out = trimesh.load(output_obj_path)
        ver_in = obj_in.vertices
        ver_out = obj_out.vertices
        ver_in = np.asarray(ver_in)
        ver_out = np.asarray(ver_out)
        diff_ver_idx = get_diff_ver_idx(ver_in, ver_out)
        obj_in, obj_out = modify_obj(obj_in,obj_out, diff_ver_idx,ver_in, ver_out)

        obj_total = combine_obj(obj_in, obj_out, diff_ver_idx)
        obj_total.export(os.path.join("./split_wound",self.filename))
        download_link = self.link+"/split-wound?file="+self.filename
        return download_link
def get_diff_ver_idx(in_ver, out_ver):
    ver_diff = np.asarray(out_ver - in_ver)
    ver_diff = np.sqrt(np.sum(ver_diff**2,axis=1))
    # ver_diff = np.abs(np.asarray(out_ver - in_ver))
    # ver_diff = np.sum(ver_diff,axis = 1)
    idx = np.where(ver_diff>6.8)
    idx = np.asarray(idx).reshape(-1,1)
    return idx
def get_diff_face(obj, idx):
    face_in = obj.faces.copy()
    id_face_diff = []
    for id in idx:
        for i in range(0, len(face_in)):
            if id in face_in[i] and i not in id_face_diff:
                id_face_diff.append(i)
    face_diff = []
    for id in id_face_diff:
        face_diff.append(face_in[id])
    face_diff = np.asarray(face_diff)
    return face_diff
def flat_face(face_diff):
    return face_diff.reshape(-1,1)

def get_total_vers_id(faces,diff_vers_idx):
    other_ver = []
    for ver in faces:
        if ver not in diff_vers_idx and ver not in other_ver:
            other_ver.append(ver)
    other_ver = np.asarray(other_ver).reshape(-1,1)
    id_ver_total = np.concatenate((diff_vers_idx,other_ver),axis = 0)
    id_ver_total = id_ver_total.reshape(-1,1)
    print(id_ver_total.shape)
    return id_ver_total
def get_vertices(ver_id, obj_ver):
    ver_total = []
    for id in ver_id:
        ver_total.append(obj_ver[id])
    ver_total = np.asarray(ver_total)
    ver_total = np.squeeze(ver_total)
    return ver_total
def create_obj(obj,id_ver_total, flat_face_diff, ver_total):
    for i in range(0,len(ver_total)):
        flat_face_diff = np.where(flat_face_diff == id_ver_total[i],i, flat_face_diff)
    face_diff = flat_face_diff.reshape(-1,3)
    obj.vertices = ver_total
    obj.faces = face_diff
    return obj
def modify_obj(input_obj,output_obj, diff_ver_idx,input_obj_vertices, output_obj_vertices):
    diff_faces = get_diff_face(input_obj, diff_ver_idx)
    flat_face_diff = flat_face(diff_faces)
    id_ver_total = get_total_vers_id(flat_face_diff, diff_ver_idx)
    input_total_ver = get_vertices(id_ver_total,input_obj_vertices)
    input_split_obj = create_obj(input_obj, id_ver_total, flat_face_diff, input_total_ver)
    output_total_ver = get_vertices(id_ver_total, output_obj_vertices)
    output_split_obj = create_obj(output_obj, id_ver_total, flat_face_diff, output_total_ver)

    return input_split_obj, output_split_obj
def combine_obj(obj_in, obj_out, diff_ver_idx):
    ver_in = np.asarray(obj_in.vertices)
    ver_out = np.asarray(obj_out.vertices)
    face_in = np.asarray(obj_in.faces)
    face_out = np.asarray(obj_out.faces)
    face_out_flatten = face_out.reshape(-1,1)
    face_out_flatten = np.where(face_out_flatten <len(diff_ver_idx), face_out_flatten + ver_out.shape[0], face_out_flatten)
    face_out = face_out_flatten.reshape(-1,3)
    ver_out = ver_out[:len(diff_ver_idx),:]
    ver_total = np.concatenate((ver_in, ver_out), axis=0)
    total_faces = np.concatenate((face_in,face_out))
    obj_in.vertices = ver_total
    obj_in.faces = total_faces
    return obj_in

