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
        self.model.cuda()
        checkpoint = torch.load(self.param.read_weight_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.init_test_mode()
        self.model.eval()
        # ip = subprocess.check_output("wget -q -O- icanhazip.com",text=True, shell=True, stderr=subprocess.STDOUT).replace("\n","")
        # self.link = "http://"+ip+":5001"
    def execute_predict(self, file_path):
        mesh = trimesh.exchange.load.load(file_path)
        ver = np.asarray(mesh.vertices)
        ver, height = self.preproccessing(ver)
        ver_out = self.predict(ver)
        ver_out = self.after_proccessing(ver_out, height)
        mesh = self.change_ver(mesh, ver_out)
        filename = file_path.split("/")[-1]
        save_file_path = self.save_predicted_obj(mesh,filename)
        # link_down_file = self.link+"/reconstructed_face/"+filename
        return mesh
    def save_predicted_obj(self, mesh, name):
        file_save_path = os.path.join("./reconstructed_face_file",name)
        mesh.export(file_save_path)
        return file_save_path
    def preproccessing(self, ver):
        ver = np.expand_dims(ver, axis=0)
        height = ver[:,:,1].mean(1)
        ver[:,:,0:3] -= ver[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(self.param.point_num,1)
        ver = torch.FloatTensor(ver).cuda()
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
        pass
def get_diff_ver_idx(ver_in, ver_out):
    ver_diff = np.abs(np.asarray(ver_out - ver_in))
    ver_diff = np.sum(ver_diff,axis = 1)
    idx = np.where(ver_diff>10)
    idx = np.asarray(idx).reshape(-1,1)
    return idx
def get_diff_face(obj, idx):
    face_in = obj.faces
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

def get_total_vers(faces,diff_vers_idx,obj_ver):
    other_ver = []
    for ver in faces:
        if ver not in diff_vers_idx and ver not in other_ver:
            other_ver.append(ver)
    other_ver = np.asarray(other_ver).reshape(-1,1)
    id_ver_total = np.concatenate((diff_vers_idx,other_ver),axis = 0)
    id_ver_total = id_ver_total.reshape(-1,1)
    print(id_ver_total.shape)
    ver_total = []
    for id in id_ver_total:
        ver_total.append(obj_ver[id])
    ver_total = np.asarray(ver_total)
    ver_total = np.squeeze(ver_total)
    return id_ver_total, ver_total
def create_obj(obj,id_ver_total, flat_face_diff, ver_total):
    for i in range(0,len(ver_total)):
        flat_face_diff = np.where(flat_face_diff == id_ver_total[i],i, flat_face_diff)
    face_diff = flat_face_diff.reshape(-1,3)
    obj.vertices = ver_total
    obj.faces = face_diff
    return obj
def modify_obj(obj, diff_ver_idx,obj_ver):
    diff_faces = get_diff_face(obj, diff_ver_idx)
    flat_face_diff = flat_face(diff_faces)
    id_ver_total, total_ver = get_total_vers(flat_face_diff, diff_ver_idx, obj_ver)
    return_obj = create_obj(obj, id_ver_total, flat_face_diff, total_ver)
    return return_obj
