import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import streamlit as st

# Streamlit 標題
st.title("Non-Circular 2D Dataset with SVM Classification")

# 生成非圓形資料集
np.random.seed(0)
num_points_per_class = 300
angle = np.linspace(0, 2 * np.pi, num_points_per_class)
r1 = 10 + 3 * np.sin(2 * angle)
r2 = 10 + 3 * np.cos(2 * angle)

x1 = r1 * np.cos(angle) + np.random.normal(0, 0.5, num_points_per_class)
y1 = r1 * np.sin(angle) + np.random.normal(0, 0.5, num_points_per_class)

x2 = r2 * np.cos(angle) + np.random.normal(0, 0.5, num_points_per_class)
y2 = r2 * np.sin(angle) + np.random.normal(0, 0.5, num_points_per_class)

X = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))
y = np.array([0] * num_points_per_class + [1] * num_points_per_class)

# 訓練 SVM 模型
svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X, y)

# 繪製資料點和決策邊界
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 100),
                     np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 100))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 繪製結果
st.write("### Data Distribution")
fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
ax.scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', label='Class 0')
ax.scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', label='Class 1')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('SVM Decision Boundary on Non-Circular Distribution')
ax.legend()
st.pyplot(fig)
