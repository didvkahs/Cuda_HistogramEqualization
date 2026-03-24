# Cuda_HistogramEqualization

---

## 1. 개요 (Overview)

본 프로젝트는 CUDA(Compute Unified Device Architecture)를 사용하여 이미지의 명암 대비(Contrast)를 자동으로 개선하는 **히스토그램 평활화(Histogram Equalization)** 알고리즘을 구현한 것입니다. 

## 2. 구현 상세

### **Histogram 구성 (병렬 빈도수 추출) >**

- Histogram 배열
    - 픽셀의 밝기값을 고려해 0 ~ 255 사이의 값들을 담을 수 있는 배열을 만들었습니다. 이렇게 설정한 이유는 밝기값을 index로 사용하기 위해서입니다.
- race condition
    - 병렬처리의 경우 동일한 메모리 값에 데이터를 작성할 경우 발생합니다.
    - 이로 인해 잘못된 값들이 들어가거나, 누적이 안되는 문제가 발생합니다.
    - 이를 해결하기 위해 atomicAdd를 사용해 해결했습니다.

### **Equalization 구현 (순위 기반 재매핑) >**

- CDF
    - 이 함수를 사용하지 않았을 경우 어두운 이미지는 0이 되버리는 문제가 발생합니다. 이를 해결하기 위해 CDF를 사용했습니다.
    - 누적합을 사용할 경우, 특정 픽셀의 값들을 평균분포에서 어느 위치에 해당하는지 확인할 수 있습니다. 이를 이용해 픽셀의 밝기값을 변경합니다.
- Histogram
    - host에서 histogram의 값들을 equalized된 값들로 변경해, 이미지 데이터에 적용했습니다.
