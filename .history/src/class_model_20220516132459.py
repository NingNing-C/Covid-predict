import torch
from torch._C import device
import torch.nn as nn
import os
import sys
import esm
from transformers.file_utils import ModelOutput
from typing import Optional
import pandas as pd
import numpy as np
dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
#print(dir)
sys.path.append(dir) 
from Struct2vec.struct2seq.struct_embed import StructEmbed
from Struct2vec.struct2seq.data import StructureDataset
from Struct2vec.experiments.utils import featurize
import os


class output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: tuple = None
    class_loss: Optional[torch.FloatTensor] = None
    embedding: Optional[torch.FloatTensor] =None


class ClassPredictionHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 droupout: float=0.1) -> None:
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_dim,hid_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(droupout,inplace=True),
            nn.Linear(hid_dim,out_dim)
            )
   
    def forward(self,pooled_output):
        value_pred = self.fc_layer(pooled_output)
        outputs = value_pred
        return(outputs)

class ValuePredictionHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 droupout: float=0.1) -> None:
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_dim,hid_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(droupout,inplace=True),
            nn.Linear(hid_dim,out_dim)
            )
   
    def forward(self,pooled_output):
        value_pred = self.fc_layer(pooled_output)
        outputs = value_pred
        return(outputs)

class covid_prediction_model(nn.Module):
    def __init__(self,
                 #class_weights,
                 jsonl_path: str=os.path.join(dir,'Struct2vec/data/covid_antibody/new_merged_all.jsonl'),
                 freeze_bert: bool=False,
                 esm_path: str="./model",
                 seq_embedding_size: int=1280,
                 stru_embedding_size: int=5,
                 stru_seq_len:int=130,
                 dropout_prob: float=0.1,
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(1111)
        #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.esm_model = esm.pretrained.load_model_and_alphabet_local(esm_path)[0].to(device)
        #device =next(self.esm_model.parameters()).device
        self.model_name = os.path.basename(esm_path)
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size+stru_embedding_size*stru_seq_len*2,512,2,dropout_prob).to(device)
        #self.predict_num = ValuePredictionHead(seq_embedding_size+stru_embedding_size*stru_seq_len*2,512,1,dropout_prob)
        self.pooling = pooling
        #self.class_weights = class_weights
        self.structEmbed = StructEmbed(node_features=128,edge_features=128,hidden_dim=128,out_dim=stru_embedding_size).to(device)
        #print("strutEmbed device:", next(self.structEmbed.parameters()).device)
        self.antibody_data = StructureDataset(jsonl_file=jsonl_path, truncate=None, max_length=500)
        X, S, mask, lengths = featurize(self.antibody_data, device=device, shuffle_fraction=0)
        self.X=X
        self.S=S
        self.mask=mask
        self.lengths=lengths

        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
        
    def forward(self,input_ids,labels):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(labels)
        #print(labels.shape)
        # if labels is not None:
        #     regre_label = labels[:,0]
        #     #class_label = labels[:,1].long()
        #     class_label = labels[:,1]
        # else:
        #     regre_label = None
        #     class_label = None
        #print("esm device:", next(self.esm_model.parameters()).device)
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        # for p in self.esm_model.parameters():
        #     print(p.shape)
        b=outputs.shape[0]
        device=outputs.device
        #print("before pooling",outputs.shape)
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
            ## outputs : [batch_size,1280]
            #print("after pooling",outputs.shape)
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        #class_logits = self.predict_class(outputs)
        #pos=torch.zeros([b,201],device=device)
        
        # for i,j in enumerate(site):
        #     pos[i,j-1]=1
        #pos [batch_size,201]
        #outputs = torch.cat((outputs,pos),dim=1)
        #outputs = torch.cat((outputs,pppl),dim=1)
        outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)
        ## outputs : [batch_size,9,1481]
        
        ## Struct_embedding : [18,130,5]
        Struct_embedding = self.structEmbed(self.X,self.S,self.lengths,self.mask)
        Struct_embedding = torch.cat((Struct_embedding[0::2,:,:],Struct_embedding[1::2,:,:]),dim=1)
        Struct_embedding = torch.flatten(Struct_embedding,start_dim=1)
        ## Struct_embedding : [9,1300]
        Struct_embedding = torch.repeat_interleave(Struct_embedding.unsqueeze(dim=0),repeats=b,dim=0)
        ## Struct_embedding : [batch_size,9,1300]
        combined_embedding = torch.cat((Struct_embedding,outputs),dim=2)
        ## combined_embedding : [batch_size,9,2580]
        class_logits = self.predict_class(combined_embedding) # [16,9,2]
        #regre_logits = regre_logits.flatten(1)
        loss = None
        #class_loss = None
        # if (class_label is not None) and (regre_label is not None):
        #     class_count_dict=pd.value_counts(class_label.detach().cpu().numpy()).to_dict()
        #     weight_list=[]
        #     for label in np.arange(0,2):
        #         if label not in class_count_dict.keys():
        #             weight_list.append(0)
        #         else:
        #             weight_list.append(1/class_count_dict[label])
        #     class_weights=torch.tensor(weight_list,dtype=torch.float)  
            #criterion_class = nn.CrossEntropyLoss(weight=class_weights.cuda())
            # criterion_class = nn.CrossEntropyLoss(weight=class_weights)
            # criterion_regre = nn.MSELoss()
            #print("logits:",logits.shape)
            #print("labels",labels.shape)
            #class_loss = criterion_class(class_logits, class_label.cuda())
        #     class_loss = criterion_class(class_logits, class_label)
        #     regre_loss = criterion_regre(regre_logits.view(-1), regre_label)
        #     loss = class_loss + regre_loss
        # if ((regre_label is None) and (class_label is not None)) or ((regre_label is None) and (class_label is not None)):
        #     ValueError("There is only one label has value")
        #print("regre_logits",regre_logits.shape)
        #print("labels",labels.shape)
        if labels is not None:
            pos_weight=torch.tensor([[0.08,0.92],[0.18,0.82],[0.06,0.94],[0.1,0.9],[0.08,0.92],[0.1,0.9],[0.05,0.95],[0.04,0.96],[0.17,0.83]],device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if labels.shape[1]==10:
                labels=labels[:,0:9,:]
            loss = criterion(class_logits,labels)
        return(output(
        loss = loss,
        logits = class_logits,
        embedding = combined_embedding
        ))
        #regre_logits = regre_logits.view(-1),
        #class_logits = class_logits))
    
class covid_prediction_model_without_GCN(nn.Module):
    def __init__(self,
                 freeze_bert: bool=False,
                 esm_path: str="./model",
                 seq_embedding_size: int=1280,
                 dropout_prob: float=0.1,
                 pos_weight:torch.tensor=torch.tensor([0.1,0.9]),
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(1111)
        #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.esm_model = esm.pretrained.load_model_and_alphabet_local(esm_path)[0].to(device)
        #device =next(self.esm_model.parameters()).device
        self.model_name = os.path.basename(esm_path)
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size,512,2,dropout_prob).to(device)
        self.pooling = pooling
        self.pos_weight=pos_weight
        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
    def forward(self,input_ids,labels):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        #outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)
        ## outputs : [batch_size,9,1280]
        ## embedding : [batch_size,9,1280]
        class_logits = self.predict_class(outputs) # [b,9,2]
        loss = None
        if labels is not None:
            # pos_weight=torch.tensor([[0.08,0.92],[0.18,0.82],[0.06,0.94],[0.1,0.9],[0.08,0.92],[0.1,0.9],[0.05,0.95],[0.04,0.96],[0.17,0.83]],device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = criterion(class_logits,labels)
        return(output(
        loss = loss,
        logits = class_logits,
        embedding = outputs
        ))


class covid_prediction_model_without_GCN_add_noise(nn.Module):
    def __init__(self,
                 freeze_bert: bool=False,
                 esm_path: str="./model",
                 seq_embedding_size: int=1280,
                 dropout_prob: float=0.1,
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(1111)
        #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.esm_model = esm.pretrained.load_model_and_alphabet_local(esm_path)[0].to(device)
        #device =next(self.esm_model.parameters()).device
        self.model_name = os.path.basename(esm_path)
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size+1300,512,2,dropout_prob).to(device)
        self.pooling = pooling
        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
    def forward(self,input_ids,labels):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)
        ## outputs : [batch_size,9,1280]
        ## embedding : [batch_size,9,1280]
        torch.manual_seed(3)
        noise=torch.randn((b,9,1300),device=device)
        combined_embedding = torch.cat((noise,outputs),dim=2)
        class_logits = self.predict_class(combined_embedding) # [b,9,2]
        loss = None
        if labels is not None:
            pos_weight=torch.tensor([[0.08,0.92],[0.18,0.82],[0.06,0.94],[0.1,0.9],[0.08,0.92],[0.1,0.9],[0.05,0.95],[0.04,0.96],[0.17,0.83]],device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if labels.shape[1]==10:
                labels=labels[:,0:9,:]
            loss = criterion(class_logits,labels)
        return(output(
        loss = loss,
        logits = class_logits,
        embedding = combined_embedding
        ))



class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2,1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range 

    def forward(self, E_idx):
        # i-j
        N_batch = E_idx.size(0)
        N_nodes = E_idx.size(1)
        N_neighbors = E_idx.size(2)
        device=E_idx.device
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).to(device)
        d = (E_idx.float() - ii).unsqueeze(-1)
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        ).to(device)
        # Grid-aligned
        # frequency = 2. * np.pi * torch.exp(
        #     -torch.linspace(
        #         np.log(self.period_range[0]), 
        #         np.log(self.period_range[1]),
        #         self.num_embeddings / 2
        #     )
        # )
        angles = d * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1).to(device)
        return E

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, features_type='full', augment_eps=0., dropout=0.1):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Feature types
        self.features_type = features_type
        self.feature_dimensions = {
            'coarse': (3, num_positional_embeddings + num_rbf + 7),
            'full': (6, num_positional_embeddings + num_rbf + 7),
            'dist': (6, num_positional_embeddings + num_rbf),
            'hbonds': (3, 2 * num_positional_embeddings),
        }

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.dropout = nn.Dropout(dropout)
        
        # Normalization and embedding
        node_in, edge_in = self.feature_dimensions[features_type]
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, self.top_k, dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

        # Debug plot KNN
        # print(E_idx[:10,:10])
        # D_simple = mask_2D * torch.zeros(D.size()).scatter(-1, E_idx, torch.ones_like(knn_D))
        # print(D_simple)
        # fig = plt.figure(figsize=(4,4))
        # ax = fig.add_subplot(111)
        # D_simple = D.data.numpy()[0,:,:]
        # plt.imshow(D_simple, aspect='equal')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('D_knn.pdf')
        # exit(0)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        # print("D device:", D.device)
        D_min, D_max, D_count = 0., 20., self.num_rbf
        device = D.device
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        # print("D_mu device:",D_mu.device)
        # print("self.num_rbf device:",self.num_rbf.device)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        # for i in range(D_count):
        #     fig = plt.figure(figsize=(4,4))
        #     ax = fig.add_subplot(111)
        #     rbf_i = RBF.data.numpy()[0,i,:,:]
        #     # rbf_i = D.data.numpy()[0,0,:,:]
        #     plt.imshow(rbf_i, aspect='equal')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig('rbf{}.pdf'.format(i))
        #     print(np.min(rbf_i), np.max(rbf_i), np.mean(rbf_i))
        # exit(0)
        return RBF

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        # Axis of rotation
        # Replace bad rotation matrices with identity
        # I = torch.eye(3).view((1,1,1,3,3))
        # I = I.expand(*(list(R.shape[:3]) + [-1,-1]))
        # det = (
        #     R[:,:,:,0,0] * (R[:,:,:,1,1] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,1])
        #     - R[:,:,:,0,1] * (R[:,:,:,1,0] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,0])
        #     + R[:,:,:,0,2] * (R[:,:,:,1,0] * R[:,:,:,2,1] - R[:,:,:,1,1] * R[:,:,:,2,0])
        # )
        # det_mask = torch.abs(det.unsqueeze(-1).unsqueeze(-1))
        # R = det_mask * R + (1 - det_mask) * I

        # DEBUG
        # https://math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        # Columns of this are in rotation plane
        # A = R - I
        # v1, v2 = A[:,:,:,:,0], A[:,:,:,:,1]
        # axis = F.normalize(torch.cross(v1, v2), dim=-1)
        return Q

    def _contacts(self, D_neighbors, E_idx, mask_neighbors, cutoff=8):
        """ Contacts """
        D_neighbors = D_neighbors.unsqueeze(-1)
        neighbor_C = mask_neighbors * (D_neighbors < cutoff).type(torch.float32)
        return neighbor_C

    def _hbonds(self, X, E_idx, mask_neighbors, eps=1E-3):
        """ Hydrogen bonds and contact map
        """
        X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

        # Virtual hydrogens
        X_atoms['C_prev'] = F.pad(X_atoms['C'][:,1:,:], (0,0,0,1), 'constant', 0)
        X_atoms['H'] = X_atoms['N'] + F.normalize(
             F.normalize(X_atoms['N'] - X_atoms['C_prev'], -1)
          +  F.normalize(X_atoms['N'] - X_atoms['CA'], -1)
        , -1)

        def _distance(X_a, X_b):
            return torch.norm(X_a[:,None,:,:] - X_b[:,:,None,:], dim=-1)

        def _inv_distance(X_a, X_b):
            return 1. / (_distance(X_a, X_b) + eps)

        # DSSP vacuum electrostatics model
        U = (0.084 * 332) * (
              _inv_distance(X_atoms['O'], X_atoms['N'])
            + _inv_distance(X_atoms['C'], X_atoms['H'])
            - _inv_distance(X_atoms['O'], X_atoms['H'])
            - _inv_distance(X_atoms['C'], X_atoms['N'])
        )

        HB = (U < -0.5).type(torch.float32)
        neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1),  E_idx)
        # print(HB)
        # HB = F.sigmoid(U)
        # U_np = U.cpu().data.numpy()
        # # plt.matshow(np.mean(U_np < -0.5, axis=0))
        # plt.matshow(HB[0,:,:])
        # plt.colorbar()
        # plt.show()
        # D_CA = _distance(X_atoms['CA'], X_atoms['CA'])
        # D_CA = D_CA.cpu().data.numpy()
        # plt.matshow(D_CA[0,:,:] < contact_D)
        # # plt.colorbar()
        # plt.show()
        # exit(0)
        return neighbor_HB

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        # Pair features

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1+eps, 1-eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0,0,1,2), 'constant', 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)

        # DEBUG: Viz [dense] pairwise orientations 
        # O = O.view(list(O.shape[:2]) + [3,3])
        # dX = X.unsqueeze(2) - X.unsqueeze(1)
        # dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        # dU = dU / torch.norm(dU, dim=-1, keepdim=True)
        # dU = (dU + 1.) / 2.
        # plt.imshow(dU.data.numpy()[0])
        # plt.show()
        # print(dX.size(), O.size(), dU.size())
        # exit(0)

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)
        
        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3,3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1)

        # DEBUG: Viz pairwise orientations
        # IMG = Q[:,:,:,:3]
        # # IMG = dU
        # dU_full = torch.zeros(X.shape[0], X.shape[1], X.shape[1], 3).scatter(
        #     2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), IMG
        # )
        # print(dU_full)
        # dU_full = (dU_full + 1.) / 2.
        # plt.imshow(dU_full.data.numpy()[0])
        # plt.show()
        # exit(0)
        # print(Q.sum(), dU.sum(), R.sum())
        return AD_features, O_features

    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)

        # print(cosD.cpu().data.numpy().flatten())
        # print(omega.sum().cpu().data.numpy().flatten())

        # Bond angle calculation
        # A = torch.acos(-(u_1 * u_0).sum(-1))

        # DEBUG: Ramachandran plot
        # x = phi.cpu().data.numpy().flatten()
        # y = psi.cpu().data.numpy().flatten()
        # plt.scatter(x * 180 / np.pi, y * 180 / np.pi, s=1, marker='.')
        # plt.xlabel('phi')
        # plt.ylabel('psi')
        # plt.axis('square')
        # plt.grid()
        # plt.axis([-180,180,-180,180])
        # plt.show()

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def forward(self, X, L, mask):
        """ Featurize coordinates as an attributed graph """

        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)

        # Pairwise features
        AD_features, O_features = self._orientations_coarse(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)

        # Pairwise embeddings
        E_positional = self.embeddings(E_idx)

        if self.features_type == 'coarse':
            # Coarse backbone features
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'hbonds':
            # Hydrogen bonds and contacts
            neighbor_HB = self._hbonds(X, E_idx, mask_neighbors)
            neighbor_C = self._contacts(D_neighbors, E_idx, mask_neighbors)
            # Dropout
            neighbor_C = self.dropout(neighbor_C)
            neighbor_HB = self.dropout(neighbor_HB)
            # Pack
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1,-1,-1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1,-1,-1, int(self.num_positional_embeddings / 2))
            E = torch.cat((E_positional, neighbor_C, neighbor_HB), -1)
        elif self.features_type == 'full':
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'dist':
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((E_positional, RBF), -1)

        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return V, E, E_idx