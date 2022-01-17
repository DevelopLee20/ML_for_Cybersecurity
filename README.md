# 도서 공부용 레포지토리

## 레퍼런스

- 제목 : 사이버 보안을 위한 머신러닝 쿡북 - 파이썬으로 구현하는 80가지 머신러닝 알고리듬
- 지은이 : 엠마누엘 츠케르만
- 원서 예제코드 : https://github.com/PacktPublishing/Machine-Learning-for-Cybersecurity-Cookbook

## 개념 노트

#
- [(2022-01-15)1일차](./2022-01-15/2022-01-15.ipynb)
    - X, y 용어 정의
        - X
            - 특성
            - Property
        - y
            - 레이블
            - Label
    - 데이터 스플릿(Data Split)
        - 기존의 데이터 불러오기
            ```python
            df = pd.read_csv('north_korea_missile_test_database.csv')
            X = df.drop("Missile Name", axis=1)
            y = df['Missile Name']
            ```
        - 훈련 데이터, 테스트 데이터 분할
            ```python
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)
            ```
        - 훈련 데이터 내에서 모델 검증용 데이터 분할
            ```python
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=31)
            ```
        - 코드 변수 설명
            ```python
            df      # 불러온 csv 데이터 파일
            X       # 특성 데이터
            y       # 레이블 데이터
            X_train # 훈련용 특성 데이터
            X_test  # 테스트용 특성 데이터
            y_train # 훈련용 레이블 데이터
            y_test  # 테스트용 레이블 데이터
            X_val   # 모델 검증용 특성 데이터
            y_val   # 모델 검증용 레이블 데이터
    - 데이터 표준화(Data Standardization)
        - 표준화 방식
            - 특성 데이터의 평균을 0, 분산을 1로 조정
        - 이유
            - 알고리듬은 <b>상대적 척도(scale)</b>에 민감하기 때문이다.
                - 상대적 척도 : 평균에서 얼마나 떨어져 있는지에 대한 정보
        - 표준화가 유용한 경우
            - <b>PE 헤더</b>를 특성으로 사용하는 경우
                - PE 헤더 : 매우 큰 값과 매우 작은 값을 가지고 있는 헤더
        - 표준화 코드
            ```python
            from sklearn.preprocessing import StandardScaler
            X_standardized = StandardScaler().fit_transform(X)
            ```
    - 인스턴스화(Instantiate)
        ```python
        class Reversy():
            def __init__(self):
                self.log = []
                self.turn = 0
                self.mark = ['O','X']

            def print_field(self):
                print("x\\y")
                
        game = Reversy() # 인스턴스화
        ```
        - 클래스 내의 객체의 변형을 정의하고 이름을 붙힌 장소(변수)에 <b>인스턴스</b>를 만드는 것
            - 인스턴스(Instance)
                - 실행중인 임의의 프로세스
                - 클래스의 현재 생성된 오브젝트
    - 주성분 분석(PCA : Principal Component Analysis)
        - 특성을 분석해 더 적은 개수의 특성으로 환원하는 것
        - <b>차원 축소(Dimension Reduction)</b>를 하는 것
            - 고차원의 데이터를 저차원의 데이터로 환원하는 것
        - 주성분 분석 코드
            ```python
            from sklearn.decomposition import PCA
            pca = PCA()
            pca.fit_transform(X_standardized)
            ```
        - 차원 축소 유효성(effectivenss) 평가
            - <b>분산을 몇 % 유지</b>하면서 목표 특성의 개수로 줄일 수 있는지 확인
                - 90%이상을 권장
            ```python
            Property_number = 40 # 목표 특성 개수
            pca.explained_variance_ratio_
            print(f'{sum(pca.explained_variance_ratio_[:Property_number])*100:6.4f} %')
            '''
            실행결과
            90.6852 %
            '''
            ```
    - 추가 알아낸 개념
        - f 인덱싱 사용시 출력 옵션 조정
            ```python
            print(f'{변수명:6.4f}')
            ```
---
