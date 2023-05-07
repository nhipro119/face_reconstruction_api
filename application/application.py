from GraphAE import graphAE as graphAE
from GraphAE import graphAE_param as Param
from GraphAE import graphAE_dataloader as Dataloader
import trimesh
import numpy as np
import torch
class Application:
    def __init__(self):
        self.param = Param.Parameters()
        self.param.batch =1
        self.param.read_config(".\\GraphAE\\0422_graphAE_dfaust\\10_conv_res.config")
        self.param.read_weight_path = ".\\GraphAE\\0422_graphAE_dfaust\\weight_10\\model_epoch0100.weight"
        self.model = graphAE.Model(param=self.param, test_mode=True)
        self.model.cuda()
        checkpoint = torch.load(self.param.read_weight_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.init_test_mode()
        self.model.eval()
    def execute_predict(self, obj):
        mesh = trimesh.exchange.load.load(obj)
        ver = np.asarray(mesh.vertices)
        ver, height = self.preproccessing(ver)
        ver_out = self.predict(ver)
        ver_out = self.after_proccessing(ver_out, height)
        mesh = self.change_ver(mesh, ver_out)
        return mesh
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
