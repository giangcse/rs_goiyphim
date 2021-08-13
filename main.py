from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.preprocessing import TransactionEncoder
from operator import itemgetter
from texttable import Texttable

import pandas as pd
import numpy as np
import json
import math
import os

# Lấy đường dẫn hiện tại
path = os.getcwd()

# Đọc csv, tạo dataframe
ratings = pd.read_csv(os.path.join(path, 'ratings.csv'))
movies = pd.read_csv(os.path.join(path, 'movies.csv'))

# Tính đánh giá trung bình của từng phim dựa trên hàm groupby của pandas thông qua ID phim
ratings = ratings.groupby(by="movieId").mean()

# Gộp đánh giá trung bình theo ID phim vào danh sách phim
movies = pd.merge(movies, ratings, on="movieId", how="inner")
# Bỏ cột userId và timestamp
movies = movies.drop(['userId', 'timestamp'], axis=1)
# Làm tròn đến chữ số thập phân số 1 cột rating
movies = movies.round(1)

# Tạo danh sách thể loại phim
m = len(movies)
tt = movies.genres
genres_list = []
item_prof = [[]]
for i in range(0, m-1):
    aa = tt.iloc[i].split('|')
    # Thêm rating vào
    # aa.append(str(movies.rating.iloc[i]))
    genres_list = genres_list + aa
    item_prof.append(aa)

# Tạo ma trận từ thể loại phim
te = TransactionEncoder()
te_ary = te.fit(item_prof).transform(item_prof)
te_ary = te_ary.astype('int')
df = pd.DataFrame(te_ary, columns=te.columns_)
movieID = movies.movieId[0:]
df.index = movieID


def prediction(id):
    # Trả về thông tin phim theo id
    print(movies.title[id-1], ": ", movies.genres[id-1],
          ' - ', movies.rating[id-1])

    # Trich vector cua phim va cac vector con lai
    this = df[id-1:id]
    other = df.drop(id-1)

    # Tính độ tương tự để gợi ý bằng cách tính cosine giữa phim được chọn và các phim còn lại
    matrix = cosine_similarity(this, other)
    max_cosine = np.where(matrix[0] == max(matrix[0]))

    # Tính số kết quả cho phim có khoang cách gần nhất
    print(len(max_cosine[0]), " result(s) for ", movies.title[id-1], '\n\n')

    # Tạo danh sách các phim được gợi ý
    recommended_list = []
    for i in max_cosine[0]:
        recommended_dict = {}
        recommended_dict = {"id": int(movies.movieId[i]), "title": movies.title[i],
                            "genres": movies.genres[i], "rating": int(movies.rating[i])}
        # Thêm vào danh sách thông tin phim dạng dict
        recommended_list.append(recommended_dict)
    # Sắp xếp theo độ giảm dần của rating
    recommended_list = sorted(
        recommended_list, key=itemgetter('rating'), reverse=True)

    return json.loads(json.dumps(recommended_list[:100]))


if __name__ == "__main__":
    id = int(input("Enter a movie ID: "))
    recommended_list = prediction(id)
    # Tạo list để hiển thị danh sách phim được gợi ý dạng table
    show_list = []
    # Header cho bảng
    show_list.append(['Title', 'Genres', 'Rating'])
    # lặp trong 10 phim đầu danh sách
    for i in recommended_list[:11]:
        show_list.append([i['title'], i['genres'], i['rating']])
    # Tạo table
    table = Texttable()
    table.add_rows(show_list)
    print(table.draw())
