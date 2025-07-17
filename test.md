# 第14回　方程式の数値解法と数値積分


## 1.方程式の数値解法

ある方程式 $f(x) = 0$の解を、数値計算を用いて近似的に求めよう。
ここでは、「2分法」と「ニュートン法」の2種類の解法を紹介します。

### 1-1. 二分法 (中間値の定理の応用)


**中間値の定理** 

 関数 $f(x)$ が区間 $[a,b]$ において連続であるとする。このとき、 $f(a)$ と $f(b)$ の間にある任意の数 $k$ に対し、 $f(c)=k$ を満たす点 $x=c$ が $(a,b)$ の中に少なくとも一つ存在する。

$\to$ $f(x)=0$の解 $x=c$は、 $f(a)f(b)<0$ を満たす区間 $(a,b)$ の中に少なくとも1つ存在する。

![二分法](Test_Newton-method2/Git_figs/二分法.jpg)


### 1-2. ニュートン法 (微分を用いた数値解法)




#### $\sin x = 0$ をニュートン法を用いて解く。

```python
import math  # 数学関数（sin, cos など）を使用するために math モジュールをインポート

# 方程式 f(x) = sin(x) を定義
def f(X):
  return math.sin(X)

# f(x) の導関数 f'(x) = cos(x) を定義（ニュートン法の計算に必要）
def df(X):
  return math.cos(X)

X0 = 1.0         # 初期値（x=1.0 から反復を開始）
EPS = 1e-7       # 許容誤差（これ以下の差になったら収束と判断）
IMAX = 50        # 最大反復回数（収束しない場合の打ち切り条件）

# ニュートン法の反復処理
for i in range(IMAX):
  # ニュートン法の公式による次の近似値の計算
  X1 = X0 - f(X0) / df(X0)

  # 現在の近似値と前回の値の差が十分小さければ収束と判定
  if abs(X1 - X0) < EPS:
    break  # 収束したのでループを抜ける
  else:
    X0 = X1  # 収束していないので現在の値を次回の初期値に更新

  # 各ステップの近似解を表示（1ステップ目は i=0 なので i+1 として表示）
  print('X(', i+1, ') =', X1)

# 反復終了後に収束したかどうかを判定して表示
if i + 1 >= IMAX:
  print('It did not converged')  # 収束せずに最大反復回数に達した場合
else:
  print('It converged')  # 所定の誤差以内で収束した場合
```
<details><summary>結果</summary>
  
```
X( 1 ) = -0.5574077246549021
X( 2 ) = 0.06593645192484066
X( 3 ) = -9.572191932508134e-05
X( 4 ) = 2.923566201412306e-13
It converged
```
</details>


### グラフの描画も追加

```python
import math                          # sin, cos などの数学関数を使うためにインポート
import matplotlib.pyplot as plt      # グラフ描画用のライブラリ
import numpy as np                   # 数値計算（等間隔配列など）に便利なライブラリ

# 解きたい方程式 f(X) = sin(X)
def f(X):
    return np.sin(X)

# f(X) の導関数（ニュートン法に必要）
def df(X):
    return np.cos(X)

X0 = 1.0       # 初期値（近くに解 π ≈ 3.14159 がある）
EPS = 1e-7     # 許容誤差（収束判定用）
IMAX = 50      # 最大反復回数（収束しないときの打ち切り）

# 各ステップの近似解 X の値を記録するリスト（収束過程の可視化用）
X_vals = [X0]

# ニュートン法による反復
for i in range(IMAX):
    X1 = X0 - f(X0) / df(X0)  # ニュートン法の更新式

    # 収束判定：新旧の値の差が十分小さければ終了
    if abs(X1 - X0) < EPS:
        X_vals.append(X1)  # 最終値を記録
        break
    else:
        X0 = X1            # 初期値を更新
        X_vals.append(X0)  # 現在値を記録

    # 各ステップの値を表示
    print('X(', i+1, ') =', X1)

# 収束結果を出力
if i+1 >= IMAX:
    print('It did not converged')  # 収束しなかった場合
else:
    print('It converged')          # 収束した場合

# --- 以下はグラフ描画 ---

# -2 から 4 の範囲で f(X) = sin(X) のグラフを作成
x = np.linspace(-2, 4, 400)
y = np.sin(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='f(X) = sin(X)')              # 関数のグラフ
plt.axhline(0, color='gray', linestyle='--')       # x軸（y=0）

# 反復点を赤い点と線で表示
plt.plot(X_vals, [math.sin(x) for x in X_vals], 'ro-', label='Newton steps')

# 各ステップ番号を点のそばに表示（文字を少し上にずらして見やすく）
for idx, x_val in enumerate(X_vals):
    plt.text(x_val, math.sin(x_val) + 0.1, f"Step {idx}", fontsize=9, ha='center')

# タイトルとラベル設定
plt.title('Newton Method for f(X) = sin(X)')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()
plt.grid(True)
plt.show()
```

<details><summary>結果</summary>
  
```
X( 1 ) = -0.5574077246549021
X( 2 ) = 0.06593645192484066
X( 3 ) = -9.572191932508134e-05
X( 4 ) = 2.923566201412306e-13
It converged
```
<img width="711" height="470" alt="result_newton-method" src="https://github.com/user-attachments/assets/7caa5e70-378c-4e6d-a902-59032ffffab7" />


</details>

#### $\frac{\pi}{2}<x_0 < \frac{3}{2}\pi$ を初期条件として計算をスタートしたらどんな結果になるでしょう？

## 2. 数値積分法 
