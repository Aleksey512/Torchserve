# Backend Torchserve Melanoma AI
### For Kernel type (9c_b7ns_1e_640_ext_15ep)

## Installation

#### 1) Upload model to ___"model_dir"___ and rename it to ___model.pth___

#### 2) Run docker compose
> docker compose up --build

#### 3) Send photo (POST)
> curl http://localhost:8080/predictions/model-name -F 'file=@photo.jpg'

#### 4) Output
```JSON
{
  "melanoma": 0.91493821144104,
  "nevus": 0.005232867784798145
}
```

## Prometeus
> http://localhost:9090

For statistics and metrics

## API

Full: [Torchserve doc](https://pytorch.org/serve/index.html)

- Check health of server (GET)
> http://localhost:8080/ping

- Check all models (GET)
> http://localhost:8081/models

- Check model instance
> http://localhost:8081/models/model-name

- Update model config (PUT)
> http://localhost:8081/models/model-name?<key1=params1>

- Send request (GET, POST)
> http://localhost:8080/predictions/<model-name>
