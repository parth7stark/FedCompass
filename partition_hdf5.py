'''
To partition the HDF5 file train_300.hdf for federated learning such that each client gets unique rows from the datasets, you can use the h5py library in Python. Here's how you can do it:
'''

import h5py
import numpy as np

def partition_hdf5_random(file_path, n_clients, rows_per_client, output_prefix):
    # Open the original HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Get the datasets - refer data_generator.py code
        h1_wave = f['/data/H1_wave']
        l1_wave = f['/data/L1_wave']
        v1_wave = f['/data/V1_wave']
        
        # Determine the total number of rows
        total_rows = l1_wave.shape[0]
        print("Total rows: ", total_rows)

        # Calculate the number of rows per client
        # rows_per_client = total_rows // n_clients
        # remainder = total_rows % n_clients

        #Hardcoding for demo
        # rows_per_client = 1000
        
        '''
        How to do random 1000 rows for each client?
        '''
        # Randomly permute the row indices
        indices = np.arange(total_rows)
        np.random.shuffle(indices)
        
        # Split the indices into n_clients sets of rows_per_client each
        for client_id in range(n_clients):
            # Calculate the start and end indices for this client's partition
            start_idx = client_id * rows_per_client
            end_idx = start_idx + rows_per_client
            client_indices = indices[start_idx:end_idx]

            # Sort the indices to avoid the HDF5 indexing error
            client_indices.sort()
            
            # Create a new HDF5 file for this client
            output_file = f"{output_prefix}_client_{client_id}.hdf5"
            with h5py.File(output_file, 'w') as out_f:
                # Create groups and datasets in the new file
                out_data_group = out_f.create_group('data')
                
                out_data_group.create_dataset(
                    'H1_wave', data=h1_wave[client_indices]
                )
                out_data_group.create_dataset(
                    'L1_wave', data=l1_wave[client_indices]
                )
                out_data_group.create_dataset(
                    'V1_wave', data=v1_wave[client_indices]
                )


        
        # start = 0
        
        # for client_id in range(n_clients):
        #     # Calculate the number of rows for this client
        #     # end = start + rows_per_client + (1 if client_id < remainder else 0)

        #     end = start + rows_per_client
            
        #     # Create a new HDF5 file for this client
        #     output_file = f"{output_prefix}_client_{client_id}.hdf5"
        #     with h5py.File(output_file, 'w') as out_f:
        #         # Create groups and datasets in the new file
        #         out_data_group = out_f.create_group('data')
                
        #         out_data_group.create_dataset(
        #             'H1_wave', data=h1_wave[start:end]
        #         )
        #         out_data_group.create_dataset(
        #             'L1_wave', data=l1_wave[start:end]
        #         )
        #         out_data_group.create_dataset(
        #             'V1_wave', data=v1_wave[start:end]
        #         )
            
        #     # Update the start index for the next client
        #     start = end

# Usage example
# partition_hdf5('train_300.hdf', n_clients=10, output_prefix='train_300_partition')

# Usage example
partition_hdf5_random('train_300.hdf', n_clients=2, rows_per_client=1000, output_prefix='train_300_partition')



'''
The code works
before partitioning:

(base) [parthpatel7173@dt-login03 combined_spin]$ h5dump -n train_300.hdf
HDF5 "train_300.hdf" {
FILE_CONTENTS {
 group      /
 group      /data
 dataset    /data/H1_wave
 dataset    /data/L1_wave
 dataset    /data/V1_wave
 }
}
(base) [parthpatel7173@dt-login03 combined_spin]$ h5dump -H -A 0 train_300.hdf
HDF5 "train_300.hdf" {
GROUP "/" {
   GROUP "data" {
      DATASET "H1_wave" {
         DATATYPE  H5T_IEEE_F32LE
         DATASPACE  SIMPLE { ( 1440000, 8192 ) / ( 1440000, 8192 ) }
      }
      DATASET "L1_wave" {
         DATATYPE  H5T_IEEE_F32LE
         DATASPACE  SIMPLE { ( 1440000, 8192 ) / ( 1440000, 8192 ) }
      }
      DATASET "V1_wave" {
         DATATYPE  H5T_IEEE_F32LE
         DATASPACE  SIMPLE { ( 1440000, 8192 ) / ( 1440000, 8192 ) }
      }
   }
}
}



After partitioning:
(base) [parthpatel7173@dt-login04 combined_spin]$ h5dump -n train_300_partition_client_0.hdf5
HDF5 "train_300_partition_client_0.hdf5" {
FILE_CONTENTS {
 group      /
 group      /data
 dataset    /data/H1_wave
 dataset    /data/L1_wave
 dataset    /data/V1_wave
 }
}
(base) [parthpatel7173@dt-login04 combined_spin]$ h5dump -H -A 0 train_300_partition_client_0.hdf5
HDF5 "train_300_partition_client_0.hdf5" {
GROUP "/" {
   GROUP "data" {
      DATASET "H1_wave" {
         DATATYPE  H5T_IEEE_F32LE
         DATASPACE  SIMPLE { ( 1000, 8192 ) / ( 1000, 8192 ) }
      }
      DATASET "L1_wave" {
         DATATYPE  H5T_IEEE_F32LE
         DATASPACE  SIMPLE { ( 1000, 8192 ) / ( 1000, 8192 ) }
      }
      DATASET "V1_wave" {
         DATATYPE  H5T_IEEE_F32LE
         DATASPACE  SIMPLE { ( 1000, 8192 ) / ( 1000, 8192 ) }
      }
   }
}
}


'''