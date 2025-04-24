# 🎨 Artist Classification using YOLOv11s

이 프로젝트는 50명의 예술작가를 분류하기 위해 YOLOv11s를 **이미지 분류(Classification)** 용으로 전환하여 학습한 분류 모델입니다.  
총 5,910장의 학습 이미지와 12,669장의 테스트 이미지가 사용되었습니다.

---

## 📁 데이터 개요

- 학습 데이터: `train.csv` (`id`, `img_path`, `artist`)
- 테스트 데이터: `test.csv` (`id`, `img_path`)
- 이미지 수:  
  - Train: **5,910장**  
  - Test: **12,669장**
- 클래스 수: **50명 작가**

> 💾 **데이터 다운로드 링크 예시**  
> [🔗 Dacon Artist ](https://dacon.io/competitions/official/236006/overview/description)  
> ※ 실제 경로는 직접 설정해주세요.

---

## 🧠 사용한 모델

- 모델: `YOLOv11s (classification mode)`
- 입력 사이즈: `224x224`
- 프레임워크: `Ultralytics YOLO`
- 환경: `Python 3.11 + macOS M3 CPU`

---

## 🏗️ 디렉토리 구조 (학습용)

yolo_dataset/
├── train/
│   ├── Vincent van Gogh/
│   ├── Claude Monet/
│   └── …
├── val/
│   ├── Vincent van Gogh/
│   ├── Claude Monet/
│   └── …

---

## 🏃‍♂️ 학습 방법

```python
from ultralytics import YOLO

model = YOLO('yolo11s-cls.pt')
model.train(
    data='yolo_dataset',
    epochs=20,
    imgsz=224,
    batch=32,
    project='art_classify',
    name='yolo11s-local'
)



⸻

🧪 테스트 예측

results = model.predict(source='./test', imgsz=224)
predicted_labels = [model.names[int(r.probs.top1)] for r in results]

test_df = pd.read_csv("test.csv")
test_df["artist"] = predicted_labels
test_df[["id", "artist"]].to_csv("sub.csv", index=False)



⸻

📈 제출 결과
	•	평가 지표: RMSLE (캐글 방식)
	•	📊 제출 점수: 0.53395

⸻

✨ 향후 개선 방향
	•	데이터 불균형 대응 (가중치/오버샘플링)
	•	앙상블 (다른 CNN 계열 모델과 결합)
	•	Confusion Matrix를 활용한 후처리

⸻
