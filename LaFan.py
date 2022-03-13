import torch
from torch.utils.data import Dataset, DataLoader

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append("..")

import numpy as np
from lafan1 import extract, utils, benchmarks


class LaFan1(Dataset):
    def __init__(self, bvh_path, train = False, seq_len = 50, offset = 10, debug = False):
        """
        Args:
            bvh_path (string): Path to the bvh files.
            seq_len (int): The max len of the sequence for interpolation.
        """
        if train:
            self.actors = ['subject1', 'subject2', 'subject3', 'subject4']
        else:
            self.actors = ['subject5']
        self.train = train
        self.seq_len = seq_len
        self.debug = debug
        if self.debug:
            self.actors = ['subject4']
        self.offset = offset
        self.data = self.load_data(bvh_path)
        self.cur_seq_length = seq_len
        

    def load_data(self, bvh_path):
        # Get test-set for windows of 65 frames, offset by 40 frames
        print('Building the data set...', self.actors)
        # X, Q, parents, contacts_l, contacts_r = extract.get_lafan1_set(\
        #                                         bvh_path, self.actors, window=self.seq_len, offset=self.offset, debug = self.debug)
        X, Q, parents, contacts_l, contacts_r = extract.get_lafan1_set(\
                                                bvh_path, self.actors, window=self.seq_len, offset=self.offset)
        # Global representation:
        q_glbl, x_glbl = utils.quat_fk(Q, X, parents)

        # if self.train:
        # Global positions stats:
        x_mean = np.mean(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        x_std = np.std(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        self.x_mean = torch.from_numpy(x_mean)
        self.x_std = torch.from_numpy(x_std)

        input_ = {}
        # The following features are inputs:
        # 1. local quaternion vector (J * 4d)
        input_['local_q'] = Q
        
        # 2. global root velocity vector (3d)
        input_['root_v'] = x_glbl[:,1:,0,:] - x_glbl[:,:-1,0,:]

        # 3. contact information vector (4d)
        input_['contact'] = np.concatenate([contacts_l, contacts_r], -1)
        
        # 4. global root position offset (?d)
        input_['root_p_offset'] = x_glbl[:,-1,0,:]

        # 5. local quaternion offset (?d)
        input_['local_q_offset'] = Q[:,-1,:,:]

        # 6. target 
        input_['target'] = Q[:,-1,:,:]

        # 7. root pos
        input_['root_p'] = x_glbl[:,:,0,:]

        # 8. X
        input_['X'] = x_glbl[:,:,:,:]

        print('Nb of sequences : {}\n'.format(X.shape[0]))

        return input_

    def __len__(self):
        return len(self.data['local_q'])

    def __getitem__(self, idx):
        idx_ = None
        if self.debug:
            idx_ = 0
        else:
            idx_ = idx
        sample = {}
        sample['local_q'] = self.data['local_q'][idx_].astype(np.float32)
        sample['root_v'] = self.data['root_v'][idx_].astype(np.float32)
        sample['contact'] = self.data['contact'][idx_].astype(np.float32)
        sample['root_p_offset'] = self.data['root_p_offset'][idx_].astype(np.float32)
        sample['local_q_offset'] = self.data['local_q_offset'][idx_].astype(np.float32)
        sample['target'] = self.data['target'][idx_].astype(np.float32)
        sample['root_p'] = self.data['root_p'][idx_].astype(np.float32)
        sample['X'] = self.data['X'][idx_].astype(np.float32)

        return sample


    def load_single_bvh_sequence(filepath, start=None, end=None):
        anim = extract.read_bvh(filepath)
        
        q = anim.quats[start:end]
        x = anim.pos[start:end]

        print(f"Sequence loaded, length {len(anim.quats)} frames.")
        
        q_glbl, x_glbl = utils.quat_fk(q, x, anim.parents) # 
        c_l, c_r = utils.extract_feet_contacts(x_glbl, [3, 4], [7, 8], velfactor=0.02)
        
        sequence = {}
        sequence['local_q'] = q
        sequence['root_v'] = x_glbl[1:, 0, :] - x_glbl[:-1, 0, :]
        sequence['contact'] = np.concatenate([c_l, c_r], -1)
        sequence['root_p_offset'] = x_glbl[-1, 0, :]
        sequence['local_q_offset'] = q[-1, :, :]
        sequence['target'] = q[-1, :, :]
        sequence['root_p'] = x_glbl[:, 0, :]
        sequence['X'] = x_glbl[:, :, :]

        print(f"Slice of length {sequence['local_q'].shape[0]} returned")

        return sequence