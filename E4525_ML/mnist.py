import numpy as np
import gzip

from skimage.measure import block_reduce
from skimage.filters import sobel_h,sobel_v

def read_images(filename):
    with gzip.open(filename, "rb") as binary_file:
        # Read the whole file at once
        data = binary_file.read(4)
        magic=int.from_bytes(data,byteorder="big")
        if (magic!=2051):
            raise exception(f"Bad binary file format, expected magic number 2051 but got {magic} instead")
        data = binary_file.read(4)
        Nimages=int.from_bytes(data,byteorder="big")
        data = binary_file.read(4)
        Nrows=int.from_bytes(data,byteorder="big")
        data = binary_file.read(4)
        Ncols=int.from_bytes(data,byteorder="big")
        #print(magic,Nimages,Nrows,Ncols)
        buffer=binary_file.read(Nimages*Ncols*Nrows)
        data=np.frombuffer(buffer,dtype="uint8",count=Nimages*Nrows*Ncols)
        #print(data.shape[0]/(Nrows*Ncols))      
        data=data.reshape((Nimages,Nrows,Ncols)).astype(np.double)/255.
        return data

def read_labels(filename):
    with gzip.open(filename, "rb") as binary_file:
        # Read the whole file at once
        data = binary_file.read(4)
        magic=int.from_bytes(data,byteorder="big")
        if (magic!=2049):
            raise exception(f"Bad binary file format, expected magic number 2049 but got {magic} instead")
        data = binary_file.read(4)
        Nimages=int.from_bytes(data,byteorder="big")   
        #print(magic,Nimages)
        buffer=binary_file.read(Nimages)
        data=np.frombuffer(buffer,dtype="uint8",count=Nimages)
       
        return data.astype(np.int)

def image_features(images,block_size,orientations):
    theta=np.linspace(0,np.pi,8,endpoint=False)
    v_x=np.cos(theta)
    v_y=np.sin(theta)
    edge_x=np.empty_like(images)
    edge_y=np.empty_like(images)
    for idx,image in enumerate(images):
        edge_x[idx]=sobel_h(image)
        edge_y[idx]=sobel_v(image)
    #print("edge_x",edge_x.shape,"edge_y",edge_y.shape)
    edges= edge_x[:,:,:,np.newaxis]*v_x+edge_y[:,:,:,np.newaxis]*v_y
    #print("edges",edges.shape)
    features=np.maximum(edges,0)#np.abs(edges) 
    block=(1,block_size,block_size,1)
    block_features=block_reduce(features,block,np.max) #np.mean)
    #print("block_features",block_features.shape)
    return block_features.reshape(len(images),-1)

class ImageFeatureModel:
    def __init__(self,base_model,size=3,orientations=8):
        self.base_model=base_model
        self.size=size
        self.orientations=orientations
    def fit(self,X,Y,X_val=None,Y_val=None):
        features=image_features(X,self.size,self.orientations)
        if not(X_val is None):
            features_val=image_features(X_val,self.size,self.orientations)
        else:
            features_val=None
        return self.base_model.fit(features,Y,features_val,Y_val)       
    def predict(self,X):
        features=image_features(X,self.size,self.orientations)
        return self.base_model.predict(features)