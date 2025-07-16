import torch.utils.data as data
import hdf5storage



class Dataset_Pro(data.Dataset):
    def __init__(self, file_path_in_f, file_path_in, file_path_out):
        super(Dataset_Pro, self).__init__()

#        读取数据
        f_in = hdf5storage.loadmat(file_path_in_f)[
            'frequency_crossroad_high_28_2700_27000_test']  # v,b,l,k,a,b,c
        H_in = hdf5storage.loadmat(file_path_in)['LiDAR_point_feature_crossroad_high_28_2700_27000_test']  # v,b,l,k,a,b,c
        H_out = hdf5storage.loadmat(file_path_out)["scatterer_gridmap_crossroad_high_28_2700_27000_test"]  # v,b,l,k,a,b,c
        print(f_in.shape,H_in.shape)

        batch = H_out.shape[0]

        f_in = f_in[:int(batch), ...]
        H_in = H_in[:int(batch), ...]
        H_out = H_out[:int(batch), ...]


        self.pred = H_out  # b,16,(48*2)
        self.prev = H_in  # b,4,(48*2)
        self.prev_f = f_in  # b,4,(48*2)

    def __getitem__(self, index):
        return self.pred[index, :], \
               self.prev[index, :], \
               self.prev_f[index, :]
#        return self.pred[index, :].float(), \
#               self.prev[index, :].float()

    def __len__(self):
        return self.pred.shape[0]