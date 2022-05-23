"""Neural Netを定義し、学習するモジュール
"""
import numpy as np
from collections import OrderedDict


class NeuralNet:
    """4層多入出力ニューラルネットワークモデルを定義するクラス
    (入力層) => (隠れ層1) => (隠れ層2) => (出力層)
    """
    def __init__(self, input_size:int, hidden1_size:int,
                 hidden2_size:int, output_size:int, lr:float,
                 weight_init_scale:float):
        """コンストラクタ

        Args:
            input_size  (int): 入力層のニューロンの数
            hidden_size (int): 隠れ層のニューロンの数
            output_size (int): 出力層のニューロンの数
            lr        (float): 学習率
            weight_init_scale (float): 重み初期化時のガウス分布のスケール
        """
        # 学習率を設定
        self.lr = lr
        # 重みとバイアスの初期化
        self.network = {}
        self.network['w1'] = weight_init_scale * np.random.randn(input_size, hidden1_size)
        self.network['b1'] = np.zeros(hidden1_size)
        self.network['w2'] = weight_init_scale * np.random.randn(hidden1_size, hidden2_size)
        self.network['b2'] = np.zeros(hidden2_size)
        self.network['w3'] = weight_init_scale * np.random.randn(hidden2_size, output_size)
        self.network['b3'] = np.zeros(output_size)
        # layer生成
        self.layers = OrderedDict()
        self.layers['affine1']   = AffineLayer(self.network['w1'], self.network['b1'])
        self.layers['relu1']     = Relu()
        self.layers['affine2']   = AffineLayer(self.network['w2'], self.network['b2'])
        self.layers['relu2']     = Relu()
        self.layers['affine3']   = AffineLayer(self.network['w3'], self.network['b3'])
        # 損失を計算するlayer生成
        self.last_layer = CrossEntropyLoss()

    def forward(self, x:'numpy.ndarray'):
        """順伝播処理

        Args:
            x (numpy.ndarray): 入力データ
            t (numpy.ndarray): 教師データ

        Returns:
            (numpy.ndarray): 出力結果
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x:'numpy.ndarray', t:'numpy.ndarray'):
        """損失を計算

        Args:
            x (numpy.ndarray): modelの出力データ
            t (numpy.ndarray): 教師データ

        Returns:
            (numpy.ndarray): 損失
        """
        return self.last_layer.forward(x, t)
    
    def backward(self):
        """逆伝播処理

        Returns:
            (dict): 各層のパラメータの勾配を持つdict
        """
        # 勾配計算
        dout = 1
        dout = self.last_layer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 勾配情報の集計
        gradients = dict()
        gradients['w1'] = self.layers['affine1'].dw
        gradients['b1'] = self.layers['affine1'].db
        gradients['w2'] = self.layers['affine2'].dw
        gradients['b2'] = self.layers['affine2'].db
        gradients['w3'] = self.layers['affine3'].dw
        gradients['b3'] = self.layers['affine3'].db
        # 勾配情報を持つdictをreturn
        return gradients
    
    def update_params(self, gradients:dict):
        """勾配降下法でネットワークパラメータを更新

        Args:
            gradients (dict): 勾配情報が格納されたdict
        """
        for key, grad in gradients.items():
            self.network[key] -= self.lr * grad
        

class AffineLayer:
    """Affine層を定義するクラス
    """
    def __init__(self, w:'numpy.ndarray', b:'numpy.ndarray'):
        """コンストラクタ

        Args:
            w (numpy.ndarray): 重み行列
            b (numpy.ndarray): バイアス行列
        """
        self.w  = w     # 重み
        self.b  = b     # バイアス
        self.x  = None  # 入力データ
        self.dw = None  # 重みの勾配
        self.db = None  # バイアスの勾配
    
    def forward(self, x:'numpy.ndarray'):
        """順伝播

        Args:
            x (numpy.ndarray): 入力データ

        Returns:
            (numpy.ndarray): 出力データ
        """
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out
    
    def backward(self, dout:'numpy.ndarray'):
        """逆伝播

        Args:
            dout (numpy.ndarray): Affine layerの出力の勾配

        Returns:
            (numpy.ndarray): Affine layerの入力の勾配
        """
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Relu:
    """ReLU層を定義するクラス
    """
    def __init__(self):
        """コンストラクタ
        """
        self.mask = None    # 0/1のnumpy配列
    
    def forward(self, x:'numpy.ndarray'):
        """順伝播

        Args:
            x (numpy.ndarray): 入力データ

        Returns:
            (numpy.ndarray): 出力データ
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout:'numpy.ndarray'):
        """逆伝播

        Args:
            dout (numpy.ndarray): ReLU layerの出力の勾配

        Returns:
            (numpy.ndarray): ReLU layerの入力の勾配
        """
        dout[self.mask] = 0
        dx = dout
        return dx


class CrossEntropyLoss:
    """CrossEntropy損失を計算する層を定義するクラス
    (softmax) => (cross_entropy_error)
    """
    def __init__(self):
        """コンストラクタ
        """
        self.loss = None    # 損失
        self.y = None       # softmaxの出力
        self.t = None       # 教師データ
    
    def softmax(self, x:'numpy.ndarray'):
        """softmax関数

        Args:
            x (numpy.ndarray): 入力データ

        Returns:
            (numpy.ndarray): 出力データ
        """
        c = np.max(x)
        exp_x = np.exp(x - c)   # オーバーフロー対策
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
        return y 
    
    def cross_entropy_error(self, y:'numpy.ndarray', t:'numpy.ndarray'):
        """CrossEntropyを計算する関数

        Args:
            y (numpy.ndarray): 入力データ
            t (numpy.ndarray): 教師データ

        Returns:
            (numpy.ndarray): 出力データ
        """
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        return -np.sum(t * np.log(y + 1e-7))  # Nan対策
    
    def forward(self, x:'numpy.ndarray', t:'numpy.ndarray'):
        """順伝播

        Args:
            x (numpy.ndarray): 入力データ
            t (numpy.ndarray): 教師データ
        
        Returns:
            (numpy.ndarray): 損失
        """
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self):
        """逆伝播

        Returns:
            (numpy.ndarray): 入力の勾配
        """
        dx = self.y - self.t
        return dx



if __name__=='__main__':
    # ニューラルネットワークモデルをインスタンス化
    nn = NeuralNet(input_size=3,
                   hidden1_size=2,
                   hidden2_size=2,
                   output_size=3,
                   lr=0.01,
                   weight_init_scale=0.1)
    
    # 入力データ作成
    x = np.array([[0.2, 0.5, 0.4]])
    # 教師データ作成
    t = np.array([[0.0, 1.0, 0.0]])
    
    # 学習
    for epoch in range(1, 10+1, 1):
        print(f"epoch:{epoch}")
        # ニューラルネットにデータを入力し、順伝播
        y = nn.forward(x)
        print(f"  出力：{y}")
        # ニューラルネットの出力データと教師データを使用して損失を計算
        loss = nn.loss(y, t)
        print(f"  損失：{loss}")
        # 誤差逆伝播で勾配計算
        gradients = nn.backward()
        print(f"  勾配：{gradients}")
        # 勾配情報を基にパラメータ(重み, バイアス)更新
        nn.update_params(gradients)
        print(f"  更新後の勾配：{nn.network}")
    
    # 学習済みモデルで推論
    y = nn.forward(x)
    print(f"\n最終出力：\n{y}")
