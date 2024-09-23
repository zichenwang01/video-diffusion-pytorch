import scipy.io
import torch
import einops

# Path to the .mat file
# mat_file_path = '/nfs/turbo/jjparkcv-turbo-large/chloehjh/data/ns/ns_1000_128_128_10_51.mat'
# mat_file_path = '/nfs/turbo/jjparkcv-turbo-large/chloehjh/data/ns-10s/ns_500_128_128_20_0.mat'
mat_file_path = '/nfs/turbo/jjparkcv-turbo-large/chloehjh/data/ns-Re1000-long/ns_500_128_128_20_12.mat'

# Load the .mat file
mat_data = scipy.io.loadmat(mat_file_path)
print("Finished loading", mat_file_path)

# Print the parameters of the .mat file
for key, value in mat_data.items():
    if key.startswith('__'):
        continue  # Skip meta entries
    print(f"Key: {key}")
    print(f"Value: {value.shape}")
    print("-" * 40)