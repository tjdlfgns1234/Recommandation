from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# MongoDB 연결 설정
client = MongoClient("mongodb://localhost:27017/")
db = client["Review"]
menu_collection = db["Data"]
restaurant_collection = db["Naver_Restaurant_Final"]

# 데이터를 로드하고 TF-IDF 벡터화를 수행
menu_data = pd.DataFrame(list(menu_collection.find()))

# 메뉴 설명 전처리 및 모든 값을 문자열로 변환
menu_data['description'] = menu_data['menu_name']
menu_data['description'] = menu_data['description'].astype(str)

# price 열을 철저히 float으로 변환
menu_data['price'] = pd.to_numeric(menu_data['price'], errors='coerce')
menu_data.dropna(subset=['price'], inplace=True)
menu_data['price'] = menu_data['price'].astype(float)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(menu_data['description'])

# 요청 데이터 모델 정의
class RecommendationRequest(BaseModel):
    food_name: str
    price: float

@app.post("/recommend_restaurants")
def recommend_restaurants(request: RecommendationRequest):
    # 사용자 입력값을 벡터화
    user_food = request.food_name
    user_price = request.price
    user_vector = vectorizer.transform([user_food])

    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # 유사도와 가격 차이를 데이터프레임에 추가
    menu_data['cosine_similarity'] = cosine_similarities
    menu_data['price_diff'] = menu_data['price'] - user_price

    # 가격 차이가 2000 이하인 경우 필터링
    filtered_data = menu_data[menu_data['price_diff'].abs() <= 2000]

    # 유사도가 높은 순으로 정렬, 유사도가 같을 경우 가격 차이가 작은 순으로 정렬
    sorted_data = filtered_data.sort_values(by=['cosine_similarity', 'price_diff'], ascending=[False, True])

    # 중복된 restaurant_id를 제거하고 상위 10개를 추출
    unique_recommendations = sorted_data.drop_duplicates(subset=['id']).head(10)
    restaurant_ids = unique_recommendations['id'].tolist()
    
    # restaurant_ids 리스트의 모든 int 타입 값을 string으로 변환
    restaurant_ids = list(map(str, restaurant_ids))

    # Naver_Restaurant_Final 컬렉션에서 해당 ID에 대한 데이터 조회
    restaurant_data = list(restaurant_collection.find({"id": {"$in": restaurant_ids}}, {"_id": 0}))
    print(restaurant_data)
    if not restaurant_data:
        raise HTTPException(status_code=404, detail="추천 결과가 없습니다.")

    return {"recommended_restaurants": restaurant_data}
