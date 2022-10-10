from tqdm import tqdm
import tarfile
import os
import shutil


def extract_aishell1(raw_tgz, to_dir):
    print(f'extracting aishell_1')
    if os.path.exists('../data/data_aishell/'):
        print(f'drop old')
        shutil.rmtree('../data/data_aishell/')
    file = tarfile.open(raw_tgz)
    file.extractall(to_dir)
    file.close()
    folder = raw_tgz[:-4] + '/wav/'
    for i in tqdm(os.listdir(folder)):
        with tarfile.open(folder + i) as reader:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(reader, folder)
        os.remove(os.path.join(folder, i))
    print(f'done')

#
# if __name__ == '__main__':
#     extract_aishell1('../data/data_aishell.tgz', '../data')
