# AIFFEL Campus Online Code Peer Review Template

* 코더 : 송반디
* 리뷰어 : 김혜원

# PRT(Peer Review Template)

* [O] **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**

  * PyTorch의 `deeplabv3_resnet101` 모델을 활용하여 고양이 이미지를 Semantic Segmentation하고, 생성된 mask를 기반으로 객체와 배경을 분리하는 코드가 작성되어 있습니다.
  * `np.where` 연산을 사용하여 고양이 영역은 원본 이미지를 유지하고, 배경 영역은 사막 배경 이미지(`sand_img`)로 대체하여 최종 합성 이미지(`result_img`)를 출력하는 전체 파이프라인이 구현되어 있습니다.
  * 이미지 로드, 모델 추론, 마스크 생성, 배경 이미지 리사이즈, 최종 합성 및 시각화 과정이 누락 없이 포함되어 있어 문제에서 요구한 최종 결과물을 확인할 수 있습니다.

* [O] **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**

  * 핵심 코드는 모델 출력 마스크를 원본 이미지 크기로 복원한 뒤, 해당 마스크를 이용해 고양이 영역과 배경 영역을 합성하는 부분이라고 생각합니다.

```python
# 마스크를 원본 크기로 Resize
output_predictions_resized = cv2.resize(
    output_predictions,
    (cat_img.shape[1], cat_img.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# 고양이 부분만 남기고 배경 적용
img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
result_img = np.where(img_mask_color == 255, cat_img, sand_img_resized)
```

```
- `INTER_NEAREST`를 사용하여 segmentation class 값이 보간 과정에서 섞이지 않도록 처리한 점이 적절합니다.
- 또한 `np.where`를 이용해 mask가 255인 영역은 원본 고양이 이미지를 유지하고, 그렇지 않은 영역은 배경 이미지로 교체하는 방식이 직관적입니다.
- 다만 `target_class_id = unique_classes[-1]` 방식은 이미지에 여러 객체가 섞여 있을 경우 의도하지 않은 클래스를 선택할 가능성이 있습니다. 따라서 고양이를 대상으로 한다면 COCO 기준 고양이 클래스 ID인 `15`를 명시하거나, 클래스 선택 이유를 주석으로 보강하면 더 안정적인 코드가 될 것 같습니다.
```

* [x] **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나 새로운 시도 또는 추가 실험을 수행해봤나요?**

  * 현재 제출된 코드에는 명시적인 에러 발생 기록, 디버깅 과정, 또는 문제 해결 과정이 자세히 기록되어 있지는 않습니다.
  * 다만 segmentation mask의 한계로 인해 경계 영역이 부정확하게 분리될 수 있다는 문제를 확인할 수 있었고, 향후 개선 방향을 고민할 여지가 있습니다.
  * 추가 실험으로는 다른 이미지에 적용해보기, 고양이 외 다른 클래스에 적용하기, 마스크 후처리 적용하기, 경계 영역 보정하기 등을 시도해볼 수 있습니다.

* [O] **4. 회고를 잘 작성했나요?**

  * 이미지 로드 → 모델 추론 → 마스크 생성 → 배경 합성 → 결과 출력으로 이어지는 전체 코드 실행 흐름이 비교적 명확하게 드러납니다.
  * Semantic Segmentation을 활용하면 객체와 배경을 분리할 수 있지만, 경계 부분에서는 완벽하지 않을 수 있다는 한계도 확인할 수 있습니다.
  * 특히 동물의 털, 귀, 발처럼 세밀한 영역은 mask가 부정확하게 생성될 수 있으므로, 추후 Matting이나 CRF, Guided Filter와 같은 후처리 기법을 적용하면 더 자연스러운 결과를 만들 수 있을 것 같습니다.

* [O] **5. 코드가 간결하고 효율적인가요?**

  * 변수명이 `cat_img`, `sand_img`, `result_img`처럼 직관적으로 작성되어 있어 코드의 목적을 이해하기 쉽습니다.
  * 전체 코드가 주피터 노트북 흐름에 맞게 순차적으로 구성되어 있어 실험 과정을 따라가기 쉽습니다.
  * 다만 현재는 하나의 이미지에 대해 순차적으로 실행되는 구조이므로, 여러 이미지에 재사용하려면 전처리, 모델 추론, 마스크 생성, 합성 과정을 함수로 분리하면 더 효율적일 것 같습니다.

# 회고

이번 코드 리뷰를 통해 DeepLabV3 기반 Semantic Segmentation을 활용하여 이미지의 객체와 배경을 분리하고, 배경을 다른 이미지로 합성하는 과정을 확인할 수 있었습니다. 모델 추론 결과로 나온 segmentation mask를 원본 이미지 크기에 맞게 resize하고, 이를 `np.where`로 활용해 픽셀 단위 합성을 수행한 점이 핵심적인 부분이었습니다.

특히 `INTER_NEAREST` 보간법을 사용한 점은 적절하다고 생각합니다. Segmentation mask는 일반 이미지처럼 연속적인 색상값이 아니라 클래스 ID를 담고 있기 때문에, bilinear interpolation을 사용하면 클래스 값이 섞일 수 있습니다. 따라서 nearest interpolation을 사용해 클래스 ID를 유지한 것은 올바른 처리입니다.

개선점으로는 `unique_classes[-1]`로 target class를 선택하는 방식보다, 고양이 클래스 ID를 명시적으로 지정하는 방법이 더 안정적이라고 생각합니다. 또한 `img_mask`를 3채널 이미지로 변환하는 대신 `img_mask[:, :, np.newaxis]`를 사용하면 메모리를 조금 더 효율적으로 사용할 수 있고, 이미지 차원 불일치 문제도 줄일 수 있습니다.

아래는 개선 예시 코드입니다.

```python
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import matplotlib.pyplot as plt

# 1. 이미지 로드 및 RGB 변환
cat_img = cv2.imread("./images/cat.jpeg")
sand_img = cv2.imread("./images/sand.jpeg")

cat_img = cv2.cvtColor(cat_img, cv2.COLOR_BGR2RGB)
sand_img = cv2.cvtColor(sand_img, cv2.COLOR_BGR2RGB)

# 2. 모델 및 전처리 설정
model = deeplabv3_resnet101(pretrained=True).eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = transform(cat_img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)["out"][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

# 3. 마스크를 원본 이미지 크기로 복원
output_predictions_resized = cv2.resize(
    output_predictions,
    (cat_img.shape[1], cat_img.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# 4. 고양이 클래스 마스크 추출
# COCO dataset 기준 cat class id는 15입니다.
CAT_CLASS_ID = 15
img_mask = (output_predictions_resized == CAT_CLASS_ID).astype(np.uint8) * 255

# 5. 배경 이미지 크기 맞추기
sand_img_resized = cv2.resize(
    sand_img,
    (cat_img.shape[1], cat_img.shape[0])
)

# 6. 마스크 차원 확장 후 이미지 합성
img_mask_3d = img_mask[:, :, np.newaxis]
result_img = np.where(img_mask_3d == 255, cat_img, sand_img_resized)

# 7. 최종 결과 시각화
plt.figure(figsize=(6, 6))
plt.imshow(result_img)
plt.axis("off")
plt.show()
```

위 코드에서는 `CAT_CLASS_ID = 15`를 명시하여 고양이 클래스만 선택하도록 수정했습니다. 또한 `img_mask[:, :, np.newaxis]`를 사용해 mask를 `(H, W, 1)` 형태로 확장했습니다. 이렇게 하면 `(H, W, 3)` 크기의 `cat_img`, `sand_img_resized`와 브로드캐스팅이 가능해져 더 안정적으로 합성을 수행할 수 있습니다.

전반적으로 프로젝트 요구사항을 충족하는 코드이며, DeepLabV3를 활용한 객체 분리와 배경 합성 과정을 이해하기 좋은 구현이라고 생각합니다.
