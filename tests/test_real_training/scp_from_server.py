import subprocess
import os 

def copy_folder_from_server(server, user, remote_folder_path, local_destination_path):
    # Ensure the local destination folder exists; if not, create it
    if not os.path.exists(local_destination_path):
        os.makedirs(local_destination_path)
        print(f"Created local directory: {local_destination_path}")
    
    # Construct the SCP command
    scp_command = f"scp -r {user}@{server}:{remote_folder_path} {local_destination_path}"

    try:
        # Run the SCP command
        result = subprocess.run(scp_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        print(f"Folder copied from {server}:{remote_folder_path} to {local_destination_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")


# server = "v-ger.cc.gatech.edu"
# user = "qxiao33"
# remote_path = "~/lfg/logdir/traj_train/MNN/pos/real/OptimLayer/run44"
# local_path = "logdir/traj_train/MNN/pos/real/OptimLayer/run44/"

# copy_folder_from_server(server, user, remote_path, local_path)



copy_folder_from_server('172.26.255.60',
                         'qingyu', 
                         '~/lfg/logdir/traj_train/MNN/pos/real_tennis/OptimLayer/run20', 
                         'logdir/traj_train/MNN/pos/real_tennis/OptimLayer/')
