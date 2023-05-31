import numpy as np
import matplotlib.pyplot as plt

def color_mapping(image):
    color_map = np.array([
        [0, 0, 0],       # 배경 (검정색)
        [128, 0, 0],     # 클래스 1 (어두운 빨강)
        [0, 128, 0],     # 클래스 2 (어두운 초록)
        [128, 128, 0],   # 클래스 3 (어두운 옥색)
        [0, 0, 128],     # 클래스 4 (어두운 파랑)
        [128, 0, 128],   # 클래스 5 (어두운 보라)
        [0, 128, 128],   # 클래스 6 (어두운 시안)
        [128, 128, 128], # 클래스 7 (회색)
        [64, 0, 0],      # 클래스 8 (진한 빨강)
        [192, 0, 0],     # 클래스 9 (밝은 빨강)
        [64, 128, 0],    # 클래스 10 (노란)
        [192, 128, 0],   # 클래스 11 (주황)
        [64, 0, 128],    # 클래스 12 (보라)
        [192, 0, 128],   # 클래스 13 (분홍)
    ])

    # 색으로 변환
    return color_map[image]

# 정확도, loss 출력
def plot_loss(history):
    plt.plot( history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('plot_loss.png')
    plt.close()

def plot_score(history):
    plt.plot(history['train_miou'], label='train_miou', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('plot_score.png')
    plt.close()
