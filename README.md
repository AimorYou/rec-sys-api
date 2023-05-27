
![Python](https://img.shields.io/badge/-Python-090909?style=for-the-badge&logo=)
![CatBoost](https://img.shields.io/badge/-CatBoost-090909?style=for-the-badge&logo=)
![LightFM](https://img.shields.io/badge/-LightFM-090909?style=for-the-badge&logo=)

# Recommendation System


This project is a part of Bachelor's graduate work at the Higher School of Economics. We implemented a recommendation 
system of sport exercises
## Services

By now, the functional services are still decomposed into two core services. Each of them can be tested, built, and deployed independently.
As you have already guessed this repository represents the first one.

### Auth service
Provides several API for user authentication and authorization with OAuth 2.0.

| Method | Path              | Description                                   | Scope |
|--------|-------------------|-----------------------------------------------|-------|
| POST   | /uaa/oauth/token  | Get new access token and refresh access token | ui    |
| POST   | /uaa/oauth/logout | Logout to revoke access token                 | ui    |


### Account service
Contains API related to creating and retrieving all user information. In this service, we are also demonstrating how we use 
user's privilege in accessing the API. For example, the access to `GET ` `/accounts` endpoint will only be allowed for user whose `READ_BASIC_INFORMATION` privilege, but the access
 to other endpoints don't require any special privilege as long as it has correct scope. Please refer to spring security docs [here](http://projects.spring.io/spring-security-oauth/docs/oauth2.html) for more details.

| Method | Path              | Description                                   | Scope |  Privilege |
|--------|-------------------|-----------------------------------------------|-------|------------|
| POST   | /accounts  | Create new account | ui    | ALL_ACCESS |
| GET    | /accounts | Get All user informations                | ui    | READ_BASIC_INFORMATION |
| GET    | /accounts/{username} | Get account with username | server | ALL_ACCESS |
| GET    | /accounts/current  | Get current account data | ui | ALL_ACCESS |

### RecSys service
The last but not the least service has two approaches inside (CatBoost and LightFM). The decision to leave 2 methods was 
due to the cold start problem for the LightFM, it was therefore decided to use CatBoost for a new user (about 5 times), 
then the LightFM should be retrained and use it.

| Method | Path              | Description                                   | Scope |  Privilege |
|--------|-------------------|-----------------------------------------------|-------|------------|
| POST   | /predict_catboost  | Receive recommendations from CatBoost | ui    | ALL_ACCESS |
| POST    | /predict_lightfm | Receive recommendations from LightFM                | ui    | ALL_ACCESS |
| POST    | /lightfm_new_user_rec/{username} | Receive recommendations from LightFM for new user | server | ALL_ACCESS |
| POST    | /add_user  | Add new user the LightFM model| ui | ALL_ACCESS |
| POST   | /add_interaction  | Add new interaction for the LightFM model| ui    | ALL_ACCESS |
| POST    | /update_lightfm_model | Update the data and the LightFM model.| ui    | ALL_ACCESS |

### User request example
```json
{
    "id": 1,
    "sex": "m",
    "aim": "weight loss",
    "age": 26,
    "height": 171,
    "weight": 86,
    "level": 8,
    "trainHands": 0,
    "trainLegs": 0,
    "trainBack": 0,
    "trainPress": 0,
    "trainChest": 0,
    "trainShoulders": 0
}
```
#### Notes
* All microservices have share database and there is no way to access the database directly from other services.
* The services in this project are using PostgreSQL for the persistent storage.
* Service-to-service communiation is done by using REST API. 



## Contributing
[Arseny Pivovarov](https://github.com/AimorYou)
[Leonid Popov](https://github.com/Lentohrastik)
