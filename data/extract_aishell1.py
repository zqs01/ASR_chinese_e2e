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
            reader.extractall(folder)
        os.remove(os.path.join(folder, i))
    print(f'done')

#
# if __name__ == '__main__':
#     extract_aishell1('../data/data_aishell.tgz', '../data')
