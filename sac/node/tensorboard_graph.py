#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Use Agg backend for headless environments
plt.switch_backend('Agg')

# 텐서보드 로그 파일 경로
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
log_dir = os.path.join(package_dir, 'runs', 'stage4')

# 디렉토리 내의 모든 파일 목록 가져오기
event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f]

# 그래프 설정
plt.figure(figsize=(10, 6))

# 각 이벤트 파일에 대해 처리
for event_file in event_files:
    # EventAccumulator 사용하여 텐서보드 로그 데이터 로드
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # 스칼라 데이터 추출 (예: 'reward/train' 스칼라 값)
    try:
        scalars = ea.scalars.Items('reward/train')

        # 스텝과 값 목록으로 분리
        steps = [scalar.step for scalar in scalars]
        values = [scalar.value for scalar in scalars]

        # 그래프 그리기
        plt.plot(steps, values, label=os.path.basename(event_file))
    except KeyError:
        print(f'Key "reward/train" not found in {event_file}')

# 그래프 라벨과 타이틀 설정
plt.xlabel('episodes')
plt.ylabel('reward')
plt.title('Training reward Over episodes')

# 레이블이 있는 경우에만 범례 추가
if plt.gca().has_data():
    plt.legend()

plt.grid(True)

# Save the plot as a PNG file
plt.savefig('training_reward_plot.png')
print('Plot saved as training_reward_plot.png')
