import cv2 as cv
import numpy as np 
import spams as sp 
import time as tm 
def toOD(Im):
	Im[Im == 0] = 1
	return -1*np.log(Im/255)
def toRGB(Im):
	return (255*np.exp(-1*Im)).astype(np.uint8)
def get_stain_matrix(Im, beta = 0.15, alpha = 1):
    Im = (Im[(Im > beta).any(axis = 1), :])
    _, V = np.linalg.eigh(np.cov(Im, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    That = np.dot(Im, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    return HE/np.linalg.norm(HE, axis=1)[:, None]

def get_concentration(Im, stain_matrix, lamda = 0.01,):
	return sp.lasso(Im.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T

def get_hematoxylin(concentration, stain_matrix, shape):
	return (255 * np.exp(-1 * np.dot(concentration[:, 0].reshape(-1,1), stain_matrix[0,:].reshape(-1,3)).reshape(shape))).astype(np.uint8)
def get_eoxin(concentration, stain_matrix, shape):
	return (255 * np.exp(-1 * np.dot(concentration[:, 1].reshape(-1,1), stain_matrix[1,:].reshape(-1,3)).reshape(shape))).astype(np.uint8)

start = tm.time()
target_image_name = "b001_0.png"
target_image = cv.imread(target_image_name)
target_image = cv.cvtColor(target_image, cv.COLOR_BGR2RGB)
target_od = toOD(target_image).reshape((-1,3))
target_stain_matrix = get_stain_matrix(target_od)
target_concentration = get_concentration(target_od, stain_matrix = target_stain_matrix)
target_max = np.percentile(target_concentration, 99, axis = 0)
with open("image_process.txt") as fp:
	names = fp.read().splitlines()
#	print(names)
	for simgname in names:
		if simgname != '':
			simgname.strip()
			simgname.strip('\n')
			source_image = cv.imread(simgname)
			source_image = cv.cvtColor(source_image, cv.COLOR_BGR2RGB)
			shape = source_image.shape
			source_od = toOD(source_image).reshape((-1,3))
			source_stain_matrix = get_stain_matrix(source_od)
			source_concentration = get_concentration(source_od, stain_matrix = source_stain_matrix)
			source_max = np.percentile(target_concentration, 99, axis = 0)
			source_concentration *= (target_max/source_max)
			source_od = np.dot(source_concentration, target_stain_matrix) 
			source_image = toRGB(source_od).reshape(shape)
			hematoxylin = get_hematoxylin(source_concentration, target_stain_matrix, shape)
			eoxin = get_eoxin(source_concentration, target_stain_matrix, shape)
			cv.imwrite("PY_TR_"+simgname, source_image)
			cv.imwrite("HE_OF_"+simgname, hematoxylin)
			cv.imwrite("EO_OF_"+simgname, eoxin)
			print(simgname + "normalized successfully")
end = tm.time()
print(end - start)