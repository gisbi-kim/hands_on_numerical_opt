#########################################################33
# 1
#########################################################33

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

########
### change this 
# 노이즈를 추가할 데이터 포인트의 비율
outlier_ratio = 0.7 # 0.9 means 90% outlier

# 차수가 k (odd) 인 다항식의 매개변수를 랜덤으로 생성
degree = 5

#########

coefficients = np.random.rand(degree + 1) - 0.5  # -0.5 ~ 0.5 범위의 랜덤 값

# x 값 범위 설정
@dataclass
class Range:
    min: float = -1.0  # 기본값을 -10으로 설정
    max: float = 1.0   # 기본값을 10으로 설정

x_range = Range()

x = np.linspace(x_range.min, x_range.max, 400)

# 다항식으로 y 값 계산
y = np.polyval(coefficients, x)

y_range = Range(min=1.2*np.min(y) - 0.1, max=1.2*np.max(y) + 0.1)

# 곡선 그리기
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'Degree {degree} Polynomial')
plt.title(f'Random Degree {degree} Polynomial')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.ylim(y_range.min, y_range.max)
plt.show()


#########################################################33
# 2
#########################################################33

# true data point k개를 곡선에서 샘플링
num_measurements = 3000

x_samples = np.linspace(x_range.min, x_range.max, num_measurements)
y_samples = np.polyval(coefficients, x_samples)

# k%의 랜덤한 포인트에만 노이즈 추가
num_noisy_points = int(num_measurements * outlier_ratio)  # 노이즈를 추가할 데이터 포인트의 수

# 노이즈를 추가할 랜덤한 인덱스 선택
noisy_indices = np.random.choice(num_measurements, num_noisy_points, replace=False)

# 모든 데이터 포인트에 대해 노이즈 0으로 초기화
noise = np.zeros(num_measurements)


# 선택되지 않은 데이터 포인트들에 더할 아주 작은 노이즈의 표준편차 정의
small_noise_std = 0.01
# 모든 데이터 포인트에 대해 아주 작은 노이즈를 먼저 추가
small_noise = np.random.normal(0, small_noise_std, num_measurements)
y_samples_noisy = y_samples + small_noise

# 선택된 데이터 포인트들에는 원래 정의된 노이즈를 추가
# to prevent the fixed-mean biased overfit, add two different modal noises.
white_noise_mean1 = 0.5
big_noise_std1 = 1.0
white_noise_mean2 = -0.2
big_noise_std2 = 0.5
y_samples_noisy[noisy_indices] += np.random.normal(white_noise_mean1, big_noise_std1, len(noisy_indices))
y_samples_noisy[noisy_indices] += np.random.normal(white_noise_mean2, big_noise_std2, len(noisy_indices))


# 샘플링한 데이터 포인트와 노이즈가 추가된 데이터 포인트 그리기
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'Degree {degree} Polynomial', linestyle='--', alpha=0.5)
plt.scatter(x_samples, y_samples_noisy, color='red', s=10, label='Sampled Data Points with Noise')
plt.title(f'Sampled Data Points from Degree {degree} Polynomial')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.ylim(y_range.min, y_range.max)
plt.grid(True)
plt.show()


#########################################################33
# 3
#########################################################33

from scipy.linalg import lstsq

# A 행렬과 b 벡터 생성
# x_samples를 기반으로 Vandermonde 행렬을 생성
A = np.vander(x_samples, N=degree+1)

# b는 노이즈가 추가된 샘플링된 데이터 포인트
b = y_samples_noisy

# 선형 시스템 Ax = b를 풀어 x(계수)를 찾음
coefficients_estimated, _, _, _ = lstsq(A, b)

# 예측된 곡선 생성
y_estimated = np.polyval(coefficients_estimated, x_samples)

# 원래 곡선(노이즈 없는), 예측된 곡선 그리기
plt.figure(figsize=(10, 6))
plt.plot(x_samples, y_samples, color='green', label='True Polynomial Curve')
plt.plot(x_samples, y_estimated, color='blue', linestyle='--', label='Estimated Polynomial Curve')
plt.scatter(x_samples, y_samples_noisy, color='red', s=10, alpha=0.5, label='Sampled Data Points with Noise')
plt.title('True vs. Estimated Polynomial Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.ylim(y_range.min, y_range.max)
plt.grid(True)
plt.show()


#########################################################33
# 4
#########################################################33

x_samples_orig = x_samples.copy()
y_samples_noisy_orig = y_samples_noisy.copy()

# Cauchy 커널의 스케일 매개변수, 데이터에 맞게 조정할 수 있습니다.
for ii, gamma in enumerate(np.linspace(0.1, 0.5, 20)):
    print('Try', ii, 'remained num datapoints:', x_samples.shape[0])

    # 초기 예측값 계산 (여기서는 단순 최소제곱 사용)
    coefficients_initial, _, _, _ = lstsq(A, b)
    y_initial_predicted = np.polyval(coefficients_initial, x_samples)

    # Cauchy 커널 가중치 계산
    residuals_initial = y_samples_noisy - y_initial_predicted
    weights_cauchy = 1 / (1 + (residuals_initial / gamma) ** 2)
    weights_cauchy = np.sqrt(weights_cauchy) # empirically much smoother converge 

    # 가중치 적용
    W_cauchy = np.diag(weights_cauchy)
    WA_cauchy = W_cauchy @ A
    Wb_cauchy = W_cauchy @ b

    # 가중치가 적용된 선형 시스템 해결
    coefficients_cauchy, _, _, _ = lstsq(WA_cauchy, Wb_cauchy)

    # 가중치가 적용된 예측된 곡선 생성
    y_cauchy_estimated = np.polyval(coefficients_cauchy, x_samples)

    # Create a single figure for the subplots with a width equal to your monitor's width
    fig, axes = plt.subplots(1, 4, figsize=(30, 6))

    # Plot 1: True vs. Estimated Polynomial Curve
    axes[0].plot(x_samples_orig, y_samples, color='green', 
                 label='True Polynomial Curve')
    axes[0].plot(x_samples_orig, y_estimated, color='blue', linestyle='--', 
                 label='Estimated Polynomial Curve')
    axes[0].scatter(x_samples_orig, y_samples_noisy_orig, color='red', s=10, alpha=0.5, 
                    label='Sampled Data Points with Noise')
    axes[0].set_title(f'[Non-robust] True vs. Estimated Polynomial Curve (outlier ratio: {outlier_ratio*100:.1f} %)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].set_ylim(y_range.min, y_range.max)
    axes[0].grid(True)

    # Plot 2: True vs. Cauchy Weighted Estimated Curve
    axes[1].plot(x_samples_orig, y_samples, color='green', label='True Polynomial Curve')
    axes[1].plot(x_samples, y_cauchy_estimated, color='blue', linestyle='--', 
                 label=f'Cauchy Weighted Estimated Curve (gamma: {gamma:.3f})')
    axes[1].scatter(x_samples, y_samples_noisy, color='red', s=10, alpha=0.5, 
                    label='Sampled Data Points with Noise')
    axes[1].set_title(f'[Robust] True vs. Cauchy Weighted Estimated Curve')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_ylim(y_range.min, y_range.max)
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Histogram of Cauchy Kernel Weights
    axes[2].hist(weights_cauchy, bins=50, color='blue', alpha=0.7)
    axes[2].set_title(f'Histogram of Cauchy Kernel Weights (gamma: {gamma:.3f})')
    axes[2].set_xlabel('Weight')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True)

    # Calculate the threshold value for the outlier_ratio
    outlier_ratio_damper_for_slow_removal = 0.25

    weight_threshold_for_outlier_removal = np.percentile(
        weights_cauchy, 
        100 * outlier_ratio * outlier_ratio_damper_for_slow_removal)
    if weight_threshold_for_outlier_removal < 0:
        weight_threshold_for_outlier_removal = 1.0
    print(f"weight_threshold_for_outlier_removal: {weight_threshold_for_outlier_removal:.3f}")

    # Plot the threshold line on the histogram
    axes[2].axvline(x=weight_threshold_for_outlier_removal, color='red', linestyle='--', 
                    label=f'weight_threshold_for_outlier_removal : {weight_threshold_for_outlier_removal * 100:.1f}%)')
    axes[2].legend()

    # Plot 4: True vs. Cauchy Weighted Estimated Curve with Low Weight Points
    # 가중치가 k 미만인 데이터 포인트의 인덱스 찾기
    low_weight_indices = np.where(weights_cauchy < weight_threshold_for_outlier_removal)[0]
    # 가중치가 k 미만인 데이터 포인트의 x 값을 추출
    x_low_weight = x_samples[low_weight_indices]
    y_low_weight_noisy = y_samples_noisy[low_weight_indices]
    axes[3].plot(x_samples_orig, y_samples, color='green', label='True Polynomial Curve')
    axes[3].plot(x_samples, y_cauchy_estimated, color='blue', linestyle='--', 
                 label=f'Cauchy Weighted Estimated Curve (gamma: {gamma:.3f})')
    axes[3].scatter(x_samples, y_samples_noisy, color='red', s=10, alpha=0.5, 
                    label='Sampled Data Points with Noise')
    axes[3].scatter(x_low_weight, y_low_weight_noisy, color='black', marker='v', alpha=0.2, 
                    label=f'Low Weight Data Points (weight < {weight_threshold_for_outlier_removal:.2f})')
    axes[3].set_title(f'[Robust] True vs. Cauchy Weighted Estimated Curve \n(Black: Low Weight Points to be removed)')
    axes[3].set_xlabel('x')
    axes[3].set_ylabel('y')
    axes[3].set_ylim(y_range.min, y_range.max)
    axes[3].legend()
    axes[3].grid(True)

    # Adjust the layout for better spacing
    plt.tight_layout()

    # Display the combined figure
    plt.show()

    if 0.999 < weight_threshold_for_outlier_removal:
        print("\nConverged. Stop the iterative-outlier-removal-based optimization.")
        print(f"The estimated value is\n {coefficients_cauchy}")
        print(f"The true answer is\n {coefficients}")
        print(f"The estimated error is\n {np.abs(coefficients_cauchy - coefficients)}")
        break
    else:
        #
        # recursive removal 
        high_weight_indices = np.where(weight_threshold_for_outlier_removal 
                                    < weights_cauchy)[0]

        # 가중치가 k 미만인 데이터 포인트의 x 값을 추출
        x_samples = x_samples[high_weight_indices]
        y_samples_noisy = y_samples_noisy[high_weight_indices]

        # A 행렬과 b 벡터 re 생성
        # x_samples를 기반으로 Vandermonde 행렬을 생성
        A = np.vander(x_samples, N=degree+1)

        # b는 노이즈가 추가된 샘플링된 데이터 포인트
        b = y_samples_noisy
