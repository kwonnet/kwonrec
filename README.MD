# KwonRec 

**KwonRec** is kwonnet’s real-time hybrid recommendation engine, combining collaborative filtering and content-based signals with continuous training from live user interactions.

<!-- To run this project -->
Drop the container
```sh 
docker compose down
```

Build and run the container
```sh
docker compose up --build -d
```

# Recommendation

To get recommendations, make request this request specifying userID

```ts
// npm install axios
import axios from "axios"
const userId = "cm9jk0ddv009cvd8i0zbosfy8"
const limit = 50
const resuslt = axios.get(`http://127.0.0.1:8003/recommend/${userId}$?limit=${limit}`)
console.log(result.data)
```

*V1* focuses on mainly on userID, postId, post content and features for collaborative filtering and content-based recommendation