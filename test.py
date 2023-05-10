import trimesh
import numpy as np
in_obj = trimesh.exchange.load.load("./CTM05853_0.obj")
out_obj = trimesh.exchange.load.load("./reconstructed_face_file/CTM05853_0.obj")
in_ver = np.asarray(in_obj.vertices)
out_ver = np.asarray(out_obj.vertices)
ver_diff = np.asarray(out_ver - in_ver)
ver_diff = np.sqrt(np.sum(ver_diff**2,axis=1))
# ver_diff = np.abs(np.asarray(out_ver - in_ver))
# ver_diff = np.sum(ver_diff,axis = 1)
idx = np.where(ver_diff>6.5)
idx = np.asarray(idx).reshape(-1,1)
(print(len(idx)))
print(max(ver_diff))