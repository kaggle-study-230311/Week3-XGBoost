# Week3-XGBoost
발표자 : 김하윤

### def encode_data(df)

idhogar 변수에 대해서 Label Encoding을 해주는 함수

이미 One-Hot Encoding되어있는 feature를 Label Encoding 형식으로 바꾸어줌

Tree model에서는 Label Encoding을 해주면 좋음

- `LabelEncoder().fit_transform`

scikit-learn을 이용해 범주형 데이터를 쉽게 수치형 데이터로 바꿀 수 있음

-one-hot encoder : 0과 1로 이루어진 다수의 열을 만듬

-label encoder : 하나의 열에 서로 다른 숫자를 입력해줌

단점 : 일괄적인 숫자 값으로 변환이 되면서 몇몇 알고리즘에는 예측 성능이 떨어지는 경우가 발생 / 변환된 숫자의 크기는 사실상 아무 의미가 없는데 모델은 어떤 의미가 있는 것으로 학습될 수 있음 → 회귀모델의 경우 성능이 떨어질 수 있음

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c74b4a43-54ec-4d2d-af41-7f1b85ab9916/Untitled.png)

[https://mizykk.tistory.com/10](https://mizykk.tistory.com/10)

### def feature_importance(forest, X_train, display_results=True)

모델 학습시에 변수 중요도를 보여주는 함수

- `forest.feature_importances_`

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c2a56fc7-387f-4538-adae-311eaf43c474/Untitled.png)

[https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

### def do_features(df)

기존의 데이터 셋에 있는 변수들을 조합하고 연산하여 새로운 변수를 만들고 이를 추가하여 학습을 진행

한 가정에 대해서는 묶어서 적용

### def convert_OHE2LE(df)

one hot encoded fields → label encoding

트리 기반의 모델링을 진행할 경우 One hot encoding의 결과에 Label encoding을 적용할 경우 모델의 성능이 향상된다는 것을 이용하는 코드

### def split_data(train, y, sample_weight=None, household=None, test_percentage=0.20, seed=None)

기존의 train test split을 진행할 때 대부분 학습 진행 전, sklearn에서 제공하는 train_test_split 패키지를 이용하여 train set 과 validation set을 나누어 학습을 진행하게 되는데

이 경우에는 전체 데이터에서 학습을 진행하는 것이 아닌 어느정도의 데이터의 손해가 일어난 생태에서 진행하게 됨

이러한 점을 보완하고자 모델링 과정 내에서 즉, 학습 도중에 분할을 진행하는 방식을 사용하여 전체 train data를 학습에 이용할 수 있도록 함

Target 데이터 값 → 불균형함 

`class_weight`라이브러리를 사용 → 불균형한 Target값에 서로 다른 가중치(weight)를 부여가능 ⇒ `y_train_weights`에 서로 다른 가중치 값이 저장

# Fit a voting classifier

여러 개의 classifier를 만들어준 뒤 voting방식을 활용

여러 개의 XGB를 합친 votingClassifier와 여러 개의 Random Forest를 합친 votingClassifier가 예측한 결과를 결합해 최종 예측을 진행

## Voting classifiers

“다수결 분류”를 뜻하는 것으로 두 가지 방법으로 분류할 수 있음

### 1. Hard Voting Classifier

여러 모델을 생성하고 그 결과를 비교함

이 때 classifier의 결과들을 집계하여 가장 많은 표를 얻는 클래스를 최종 예측값으로 정하는 것

다음 그림과 같이 최종 결과를

1로 예측한 모델 → 3개

2로 예측한 모델 → 1개

이므로 Hard Voting Classifier의 최종 결과는 1이 됨

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/133f870d-9e36-417c-9354-b494b9b075d5/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5add445e-96e0-4310-9b2c-68d21bc8f986/Untitled.png)

### 2. Soft Voting Classifier

앙상블에 사용되는 모든 분류기가 클래스의 확률을 예측할 수 있을 때 사용

각 class 별로 모델을 학습시키고 모델들이 예측한 probability를 합산하여 가장 높은 class 선택

각 분류기의 예측을 평균 내어 확률이 가장 높은 클래스로 예측 (가중치 투표)

아래와 같이 예측 확률에 대한 평균이 높게 나오는 클래스를 최종 예측 클래스로 정하게 됨

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fb45bb8d-d40d-4b03-af08-d242cfc6839c/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6b2fb21a-498c-49c1-993b-8416b27cb069/Untitled.png)

[https://nonmeyet.tistory.com/entry/Python-Voting-Classifiers다수결-분류의-정의와-구현](https://nonmeyet.tistory.com/entry/Python-Voting-Classifiers%EB%8B%A4%EC%88%98%EA%B2%B0-%EB%B6%84%EB%A5%98%EC%9D%98-%EC%A0%95%EC%9D%98%EC%99%80-%EA%B5%AC%ED%98%84)

## Estimator

사이킷런에서는 분류 알고리즘을 구현한 클래스를 Classifier로, 회귀 알고리즘을 구현한 클래스를 Regressor로 지칭하고, 이 둘을 합쳐서 Estimator 클래스라고 부름 (지도학습의 모든 알고리즘을 구현한 클래스를 통칭함)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/73779ba5-2258-4e4a-88ac-11fff547bacf/Untitled.png)

### estimator의 fit(), predict()

cross_val_score()와 같은 evaluation 함수, GridSearchCV와 같은 하이퍼 파라미터 튜닝을 지원하는 클래스의 경우 이 Estimator를 인자로 받음

인자로 받은 Estimator에 대해서 cross_val_score(), GridSearchCV.fit() 함수 내에서 Estimator의 fit()과 predict()를 호출해서 평가를 하거나 하이퍼 파라미터 튜닝을 수행함

### 비지도학습의 fit, transform

사이킷런 비지도 학습의 차원축소, 클러스터링, 피처 추출 등을 구현한 클래스 역시 fit() 과 transform()을 적용하지만, 이것은 지도학습에서의 fit,transform()과 다른 의미임

비지도 학습에서의 fit() → 입력 데이터의 형태에 맞춰 데이터를 변환하기 위한 사전 구조를 맞추는 작업

transform → fit 이후 입력 데이터의 차원 변환, 클러스터랑, 피처 추출 등의 실제 작업을 수행

## **XGBoost (Extreme Gradient Boosting)**

부스트래핑(Boostrapping) 방식의 앙상블 알고리즘

이전 모델의 오류를 순차적으로 보완해나가는 방식으로 모델을 형성하는데,

이전 모델에서의 실제값과 예측값의 오차(loss)를 training data에 투입하고 gradient를 이용하여 오류를 보완하는 방식을 사용

균형 트리 분할 방식(level-wise)으로 모델을 학습하여 대칭적 트리를 만듬

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/57e87896-1867-4cd4-8de0-3a92179332a6/Untitled.png)

1번 모델 : Y = w1 * H(x) + error1

1번 모델 오차(loss)의 보완 : error1 = w2 * G(x) + error2

2번 모델 : Y = w1 * H(x) + w2 * G(x) + error2

2번 모델 오차의 보완 : error2 = w3 * M(x) + error3

3번 모델 : Y = w1 * H(x) + w2 * G(x) + w3 * M(x) + error3

3번 모델 오차의 보완 : error3 = w4 * K(x) + error4

## Random forest

결정 트리를 기반으로 하는 알고리즘

여러 개의 결정 트리 분류기가 각자의 데이터를 샘플링하여 학습을 수행한 후에 최종적으로 예측 결정
