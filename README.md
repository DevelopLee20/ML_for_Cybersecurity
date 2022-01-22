# 도서 공부용 레포지토리

## 레퍼런스

- 제목 : 사이버 보안을 위한 머신러닝 쿡북 - 파이썬으로 구현하는 80가지 머신러닝 알고리듬
- 지은이 : 엠마누엘 츠케르만
- 원서 예제코드 : https://github.com/PacktPublishing/Machine-Learning-for-Cybersecurity-Cookbook

## 개념 노트

#
- [(2022-01-15) 1일차](./note/2022-01-15.ipynb)
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
            - 매우 크거나 작은 값을 특성으로 사용하는 경우
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

#
- [(2022-01-19) 2일차](./note/2022-01-19.ipynb)
    - 마르코프 연쇄
        - 시간에 따른 계의 상태 변화
            - 상태 변화의 종류
                - 같은 상태 : 상태가 변화하지 않음을 의미
                - <b>전이</b> 상태 : 상태가 변화함을 의미
        - 메인 소스 코드
            ```python
            import markovify
            from itertools import chain
            review_subset = df['content'][0:N]
            text = ''.join(chain.from_iterable(review_subset))
            markov_chain_model = markovify.Text(text)
            ```
            - chain.from_iterable 메서드(method) 코드
                ```python
                def from_iterable(iterables):
                    # chain.from_iterable(['ABC', 'DEF']) --> A B C D E F
                    for it in iterables:
                        for element in it:
                            yield element
                ```
        - 마르코프 연쇄 모델 임의의 문장 생성
            ```python
            import markovify
            markov_chain_model = markovify.Text(text)
            markov_chain_model.make_sentence() # 문장 생성

            sentence_length = 140 # 생성할 문장의 최대 길이
            print(markov_chain_model.make_short_sentence(sentence_length)) # 문장 생성
            ```
    - 제너레이터(Generator)
        - 함수 안에서 <b>yield</b> 키워드를 사용
        - 순서있는 <b>객체 생성</b> 가능
        - 같은 제너레이터로 생성한 객체끼리 비교시 <b>False</b> 출력
            - 이유 : 생성된 객체가 서로 다르기 때문이다.
        - 소스 코드
            ```python
            def generate():
                yield 1
                yield 3
                yield 2
        
            for i in generate():
                print(i, end=' ')

            print()

            a = generate()
            for i in range(3):
                print(next(a), end=' ')
            
            '''
            실행결과
            1 3 2
            1 3 2
            '''
            ```
    - 군집화(Clustering)
        - <b>유사성(similarity)</b>를 기반으로 데이터의 일부를 <b>집단(group)</b>으로 만드는 비지도 학습 알고리즘
        - 사이버 보안에서 유용하게 사용된다.
            - 정상(normal), 비정상(anomalous) 네트워크 활동을 구별
            - 악성코드를 분류
                - 정상 코드 : benign
                - 악성 코드 : malware
        - K-평균(K-means) 알고리즘
            - 라벨(Label)이 없는 특성들에 라벨(Label)을 붙여준다.
            - 소스 코드
                ```python
                from sklearn.cluster import KMeans
                label_num = len(set(y) # 분류할 라벨(Label)의 개수
                estimator = KMeans(n_clusters=label_num))
                estimator.fit(X)
                ```
                - n_clusters 속성
                    - 분류할 <b>클러스터(Cluster)</b>의 개수
                        - 클러스터(Cluster) : 집합을 의미
    - 판다스 데이터프레임 카테고리 타입
        - Pandas DataFrame category type
            ```python
            y_pred = estimator.predict(X)
            df["pred"] = y_pred
            df["pred"] = df["pred"].astype("category") # category type
            df
            ```
        - 판다스에서 숫자가 아닌 값들은 NaN(Not a Number) 또는 category type가 됨
            - 기본 값은 'NaN', astype()으로 설정시 'category' 타입 사용 가능함
    - plotly
        - 3D 그래프 출력 라이브러리
        - 3D 이미지를 출력해주기 때문에 이해하기 쉬움
        - 소스 코드
            ```python
            import plotly.express as px
            fig = px.scatter_3d(
                datefame,
                x="X_name",
                y="Y_name",
                z="Z_name",
                color="Label_color"
            )
            fig.show()
            ```
    - PE헤더(PE Header)
        - PE : Portable Executable File Format
        - 다른 컴퓨터에 파일을 옮겨도 실행 할 수 있는 파일 확장자의 집합
            - 예시 : exe, dll, obj 등등

#
- [(2022-01-21) 3일차](./note/2022-01-21.ipynb)
    - 라이브러리 사용 머신러닝
        - XGBoost
            - 악성코드 탐지에 유리
            - 이진트리를 생성해 노드를 겹쳐 최종 값을 뽑아내는 형태
            - 학습 코드
                ```python
                from xgboost import XGBClassifier
                XGB_model_instance = XGBClassifier() # XGB 인스턴스
                XGB_model_instance.fit(X_train, y_train)
                ```
        - Statsmodels
            - 시계열 분석
                - 시계열 : 연속적인 시간에서 얻은 값들로 이뤄진 수열
                    - 예시 : 주식시장의 가격
                - 사이버 공격(Cyber Attack)을 예측하는데 유용
                - 학습 코드
                    ```python
                    from statsmodels.tsa.ar_model import AR
                    model = AR(X_train)
                    model_fit = model.fit()
                    ```
        - Moving Average
            - 이동 평균 MA
                - 학습 코드
                    ```python
                    from statsmodels.tsa.arima_model import ARMA
                    model = ARMA(X_train, order=(0,1))
                    model_fit = model.fit(disp=False)
                    ```
        - Simple Exponential Smoothing
            - 단순 지수 평활 SES
                - 학습 코드
                    ```python
                    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                    model = SimpleExpSmoothing(time_series)
                    model_fit = model.fit()
                    ```
        - 학습된 모델 매개변수 확인
            ```python
            model_fit.params
            ```
        - 학습된 모델 예측값 확인
            ```python
            model_fit.predict(100,100) # 100~100 데이터 예측
            ```
        - 학습된 모델 성능 평가
            ```python
            from sklearn.metrics import accuracy_score
            y_test_pred = model_fit.predict(X_test)
            accuracy_score(y_test, y_test_pred) # 테스트라벨, 예측라벨
            ```
    - plotly 라이브러리 그래프 출력
        - 선형 그래프
            ```python
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Scatter(x=X_train, y=y_train))
            fig.show()
            ```

#
- [(2022-01-22) 4일차](./note/2022-01-22.ipynb)
    - 격리 포레스트(Isolation Forest)
        - 비정상 행위(Anomaly Detection)를 탐지
        - 데이터셋에서 기대 패턴(Expected Pattern)에 부합하지 않는 사건(Event)을 식별
        - 특이점(Outlier)을 격리하기 쉽지만 정상 데이터 점(Normal Data Point)을 설명하는 것은 어려움
        - 알고리즘에 비정상 데이터(Anormalous Data)에 관한 어떠한 정보도 주지 않음
        - 오탐지(False Position)의 수가 상당히 발생할 수 있음
            - 분류기를 미세하게 조정해야함
        - 학습 코드
            ```python
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest()
            clf.fit(X_train) # DataFrame 사용 가능
            ```
    - 랜덤 시드(Random Seed)
        - 랜덤 시드 생성
            ```python
            np.random.RandomState(12)
            ```
            - 인스턴스화 후 사용함
        - 랜덤 배열 생성
            - 기본 사용법
                ```python
                rnd = np.random.RandomState(12)
                rnd.randn(Row,Column)
                ```
                - Row x Column 사이즈의 랜덤 배열이 생성됨
            - 최대값 최소값 지정
                ```python
                rnd = np.random.RandomState(12)
                rnd.uniform(low=-5, high=5, size=(2,2))
                ```
                - low : 최소값
                - high : 최대값
                - size : 배열크기
    - 넘파이(Numpy)
        - 두 배열 합치기
            ```python
            np.r_[Array1, Array2]
            ```
            - 배열1 아래에 배열2가 연결됨
    - plotly
        - 산점도(Scatter) 그래프 출력
            ```python
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=X_train['x'],
                    y=X_train['y'],
                    mode='markers',
                    marker=dict(
                        color='color',
                        line=dict(
                            color='color',
                            width=2
                        )
                    ),
                    name='name'
                )
            )
            fig.show()
            ```
            - 필요한 산점도 종류만큼 add_trace를 사용
            - plotly docs에 더 효율적인 방식이 있음
    - 표 형식 출력
        - 소스 코드
            ```python
            from tabulate import tabulate
            tabulate(
                tabular_data=df,
                headers=columns_name,
                tablefmt='psql'
            )
            ```
            - tabular_data : 데이터프레임
            - headers : 열 이름
            - tablefmt : 테이블포멧
                - psql : PostgreSQL 테이블 포멧의 일종
    - 추가 알아낸 개념
        - 파이썬 연산자(Python Operator)
            ```python
            print(10+1)
            print(10-1)
            print(10+-1)
            print(10-+-1)
            '''
            실행결과
            11
            9
            9
            11
            '''
            ```
---
