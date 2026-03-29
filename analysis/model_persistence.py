"""
模型持久化 - v5专用版本
保存和加载v5的LR+KNN+GB模型
"""
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
CONFIG_FILE = os.path.join(MODEL_DIR, 'model_config.json')


def save_v5_model(models_dict, metadata):
    """保存v5模型"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 提取参数
    serializable = {}
    
    for name, model in models_dict.items():
        if name == 'norm':
            serializable['norm_mean'] = model[0].tolist()
            serializable['norm_std'] = model[1].tolist()
        elif name == 'weights':
            serializable['weights'] = model.tolist() if hasattr(model, 'tolist') else list(model)
        else:
            # 提取模型参数
            params = {}
            if hasattr(model, 'weights') and model.weights is not None:
                params['weights'] = model.weights.tolist()
            if hasattr(model, 'bias') and model.bias is not None:
                params['bias'] = float(model.bias)
            if hasattr(model, 'mean') and model.mean is not None:
                params['mean'] = model.mean.tolist()
            if hasattr(model, 'std') and model.std is not None:
                params['std'] = model.std.tolist()
            if hasattr(model, 'alpha'):
                params['alpha'] = model.alpha
            if hasattr(model, 'k'):
                params['k'] = model.k
            if hasattr(model, 'X'):
                params['X'] = model.X.tolist()
            if hasattr(model, 'y'):
                params['y'] = model.y.tolist()
            if hasattr(model, 'init_pred'):
                params['init_pred'] = float(model.init_pred)
            if hasattr(model, 'trees'):
                params['trees'] = model.trees
            if hasattr(model, 'lr'):
                params['lr'] = model.lr
            serializable[name] = params
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(serializable, f)
    
    # 保存元数据
    meta_save = {}
    for k, v in metadata.items():
        if isinstance(v, (np.floating, np.integer)):
            meta_save[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, list):
            meta_save[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
        elif isinstance(v, np.ndarray):
            meta_save[k] = v.tolist()
        else:
            meta_save[k] = v
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(meta_save, f, indent=2, ensure_ascii=False)
    
    print(f"模型已保存到: {MODEL_FILE}")


def load_v5_model():
    """加载v5模型"""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(CONFIG_FILE):
        return None, None
    
    with open(MODEL_FILE, 'rb') as f:
        data = pickle.load(f)
    
    with open(CONFIG_FILE, 'r') as f:
        metadata = json.load(f)
    
    # 重建归一化参数
    mean = np.array(data['norm_mean'])
    std = np.array(data['norm_std'])
    
    # 重建v5模型
    from analysis.ml_model_v5 import LinearRegression, KNN, GradientBoosting
    
    models = {}
    models['norm'] = (mean, std)
    
    if 'lr' in data:
        lr = LinearRegression(alpha=data['lr'].get('alpha', 1.0))
        lr.weights = np.array(data['lr']['weights'])
        lr.bias = data['lr']['bias']
        lr.mean = np.array(data['lr']['mean'])
        lr.std = np.array(data['lr']['std'])
        models['lr'] = lr
    
    if 'knn' in data:
        knn = KNN(k=data['knn'].get('k', 5))
        knn.X = np.array(data['knn'].get('X', []))
        knn.y = np.array(data['knn'].get('y', []))
        models['knn'] = knn
    
    if 'gb' in data:
        gb = GradientBoosting()
        gb.init_pred = data['gb'].get('init_pred', np.mean(data['knn'].get('y', [100])))
        gb.trees = data['gb'].get('trees', [])
        gb.lr = data['gb'].get('lr', 0.08)
        models['gb'] = gb
    
    if 'weights' in data:
        models['weights'] = np.array(data['weights'])
    
    print(f"模型已从文件加载: {MODEL_FILE}")
    
    return models, metadata


def model_exists():
    return os.path.exists(MODEL_FILE) and os.path.exists(CONFIG_FILE)


def get_model_age_days():
    if not os.path.exists(MODEL_FILE):
        return None
    mtime = os.path.getmtime(MODEL_FILE)
    import time
    return (time.time() - mtime) / 86400