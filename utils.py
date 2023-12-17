import numpy as np
import json
from pathlib import Path
import psutil
import torch
import gc
import os
import sys
import random
import gzip
import importlib

def capitalize(s):
    return s[:1].upper() + s[1:]

def load_class(file_name,folder='./models'):
    '''Returns class from file (assume same name as file name)'''
    file_name,module = get_module(Path(folder),file_name)
    globals()[file_name]=module
    agent_class=  getattr(globals()[file_name],capitalize(file_name))
    return agent_class

def get_module(path,file_name):
    '''get module obj from a directory (non recursive)'''
    module_spec = importlib.util.spec_from_file_location(file_name, 
        ensure_suffix(path / file_name,'.py'))
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error Occurred: File \"{file_name}\" likely does not exist in directory \"{path}\".")
        print("Traceback:")
        print(e)
        raise e
    return file_name,module


def resetSeeds():
    '''Ensure anything running this behaves the exact same, even with multiple notebook cell runs.'''
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

def memDebugger():
    '''Prints information about memory, CPU usage, and existing tensor objects'''
    print("CPU Usage (%):", psutil.cpu_percent())
    print(psutil.virtual_memory())
    pid = os.getpid()
    py = psutil.Process(pid)
    memory = py.memory_info()[0] / (2**30) #assume base 2
    print("Memory (GB):",memory)
    print("=========================")
    print("Torch objects:")
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj),obj.size(),sys.getsizeof(obj))


def json_reformatter(obj):
    '''Used to convert numpy to list when saving json file via dump()'''
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj,np.float32):
        return obj.astype(float)
    raise TypeError(f'{type(obj)} could not be reformatted.')


def ensure_suffix(path,suffix):
    if Path(path).suffix == suffix:
        return path
    else:
        if isinstance(path,Path):
            return path.parent / (path.name+suffix)
        elif isinstance(path,str):
            return path+suffix
        else:
            raise TypeError(f'object {path} is type {type(path)}.')

def save_json(data,folder,file_name: str,gz=True):
    '''saves json to file'''

    if gz:
        save_path = ensure_suffix(folder / file_name,'.gz')
    else:
        save_path = ensure_suffix(folder / file_name,'.json')

    
    ensure_folder_exists(folder)
    if gz:
        with gzip.open(save_path, 'wt', encoding='utf-8') as fp:
            json.dump(data, fp,indent = 0,default=json_reformatter)
    else:
        with open(save_path, "w") as fp:
            json.dump(data,fp, indent = 0,default=json_reformatter)

def read_json(path,gz=True):
    if gz:
        path = ensure_suffix(path,'.gz')
        with gzip.open(path, 'rt', encoding='utf-8') as fp:
            data = json.load(fp)
    else:
        path = ensure_suffix(path,'.json')
        with open(path, 'rt', encoding='utf-8') as fp:
            data = json.load(fp)
    return data


def save_np(data,folder,file_name: str,gz=True):
    '''saves np to file'''

    if gz:
        save_path = ensure_suffix(folder / file_name,'.gz')
    else:
        save_path = ensure_suffix(folder / file_name,'.npy')

    ensure_folder_exists(folder)
    if gz:
        f = gzip.GzipFile(save_path, 'w')
        np.save(f,data)
        f.close()
        #with gzip.open(save_path, 'wt', encoding='utf-8') as fp:
        #    np.save(fp,data)
    else:
        with open(save_path, "wb") as fp:
            np.save(fp,data)

def read_np(path,gz=True):
    if gz:
        path = ensure_suffix(path,'.gz')
        f = gzip.GzipFile(path, 'r')
        data = np.load(f)
        f.close()
        #with gzip.open(path, 'rt', encoding='utf-8') as fp:
        #    data = np.load(fp)
    else:
        path = ensure_suffix(path,'.npy')
        with open(path, 'rb', encoding='utf-8') as fp:
            data = np.load(fp)
    return data



def save_fig(fig,folder,file_name,suffix='.jpg'):
    '''saves pyplot fig image'''
    ensure_folder_exists(folder)
    save_path = ensure_suffix(folder / file_name,suffix)
    fig.savefig(save_path)

def all_files_exist(files):
    return all([os.path.exists(f) for f in files])

def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)