# clone: 同じパラメータを持つ新しい推定量を構築するメソッド
from sklearn.base import clone
# combinations: 全ての組み合わせのタプルのジェネレータを生成する
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS():
    """
    逐次後退選択(sequential backward selection)を実行するクラス
    """

    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring              # 特徴量を評価する指標
        self.estimator = clone(estimator)   # 推定器
        self.k_features = k_features        # 選択する特徴量の個数
        self.test_size = test_size          # テストデータの割合
        self.random_state = random_state    # 乱数種を固定する

    def fit(self, X, y):
        # トレーニングデータとテストデータに分割する
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
        # 全ての特徴量の個数,列のインデックス
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))  # range(dim)からタプルに変換
        self.subsets_ = [self.indices_]    # さらにlistにする
        # 全ての特徴量を用いてスコアを算出
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        # スコアを格納
        self.scores_ = [score]

        # 指定した特徴の個数になるまで処理を反復
        while dim > self.k_features:
            # スコア，インデックスを格納するためのリストを生成
            scores = []
            subsets = []

            # 特徴の部分集合を表す列インデックスの組み合わせごとに処理を反復
            for p in combinations(self.indices_, r=dim - 1):  # r:抽出する要素数
                # スコアを算出して格納
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                # 特徴量の部分集合を表す列インデックスのリストを格納
                subsets.append(p)

            best = np.argmax(scores)
            # 最良のスコアとなる列インデックスを抽出して格納
            self.indices_ = subsets[best]
            # 最良のスコアとなる列インデックスの移り変わりを見るために格納
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        # 最後に格納したスコア
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        # 抽出した特徴量を返す
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # 指定された列番号indicesの特徴量を抽出してモデルに適合
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        # 真のクラスラベルと予測値を用いてスコアを算出
        score = self.scoring(y_test, y_pred)
        return score
